import io
import os
import faiss
import numpy as np
import pdfplumber
import pytesseract
import firebase_admin
import threading
import time

from firebase_admin import credentials, firestore
from datetime import datetime
from PIL import Image
from docx import Document
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from app import save_feedback, llm, generate_followups
from data_loader import download_files

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= BACKGROUND DOWNLOAD =================

def load_data_background():
    print("🚀 Starting background data download...")
    download_files()

threading.Thread(target=load_data_background).start()

# ================= FIREBASE =================

import json

firebase_creds = os.getenv("FIREBASE_CREDENTIALS")

if not firebase_creds:
    raise Exception("FIREBASE_CREDENTIALS not set")

cred_dict = json.loads(firebase_creds)

cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)

db = firestore.client()

print("✅ Firebase connected successfully")

# ================= MEMORY =================

conversation_memory = {}

def update_memory(session_id, role, message):

    if session_id not in conversation_memory:
        conversation_memory[session_id] = []

    conversation_memory[session_id].append({
        "role": role,
        "content": message
    })

    conversation_memory[session_id] = conversation_memory[session_id][-10:]


def save_to_firebase(session_id, role, message):

    session_ref = db.collection("sessions").document(session_id)

    session_ref.set({
        "session_id": session_id,
        "created_at": datetime.utcnow()
    }, merge=True)

    session_ref.collection("messages").add({
        "role": role,
        "text": message,
        "timestamp": datetime.utcnow()
    })


# ================= MODELS =================

embed_model = None
reranker = None

UPLOAD_DIR = "uploaded_indexes"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ================= PATHS =================

DATA_DIR = "data"

INDEX_PATH = os.path.join(DATA_DIR, "kombucha_index.faiss")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.npy")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.npy")

# ================= FIXED: GLOBAL PLACEHOLDERS =================

research_index = None
research_chunks = None
research_metadata = None
bm25 = None


# ================= FIXED: STARTUP LOADER =================

@app.on_event("startup")
def load_data():
    import threading

    def background_loader():
        global research_index, research_chunks, research_metadata, bm25, embed_model, reranker

        print("🚀 Background loading started...")

        # ---------- WAIT FOR FILES ----------
        while not (
            os.path.exists(INDEX_PATH) and
            os.path.exists(CHUNKS_PATH) and
            os.path.exists(METADATA_PATH)
        ):
            print("⏳ Waiting for FAISS files...")
            time.sleep(5)

        print("✅ Files ready. Loading FAISS...")

        research_index = faiss.read_index(INDEX_PATH)
        research_chunks = np.load(CHUNKS_PATH, allow_pickle=True).tolist()
        research_metadata = np.load(METADATA_PATH, allow_pickle=True).tolist()

        tokenized_corpus = [chunk.split() for chunk in research_chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        print("✅ FAISS + BM25 ready")

        # ---------- LOAD MODELS ----------
        print("⏳ Loading embedding models...")

        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        print("✅ Models loaded")

    # 🔥 RUN IN BACKGROUND (IMPORTANT)
    threading.Thread(target=background_loader).start()

# ================= REQUEST MODELS =================

class ChatRequest(BaseModel):
    message: str
    session_id: str


class FeedbackRequest(BaseModel):
    query: str
    response: str
    feedback: str


# ================= TEXT CHUNKING =================

def chunk_text(text, chunk_size=800, overlap=150):

    chunks = []
    start = 0

    while start < len(text):

        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


# ================= MULTI QUERY =================

def generate_search_queries(question):

    prompt = f"""
Generate 3 scientific search queries related to the question.

Question:
{question}

Return only queries separated by newline.
"""

    try:

        queries = llm.generate(prompt, temperature=0.3)

        query_list = [q.strip() for q in queries.split("\n") if q.strip()]
        query_list.insert(0, question)

        return query_list[:4]

    except:
        return [question]


# ================= SELF REFLECTION =================

def verify_answer(question, context, answer):

    prompt = f"""
You are verifying an AI answer.

Question:
{question}

Context:
{context}

Answer:
{answer}

If the answer contains unsupported claims, rewrite it so it ONLY contains information supported by the context.

Return ONLY the corrected answer.
"""

    try:
        improved = llm.generate(prompt, temperature=0.2)
        return improved
    except:
        return answer


# ================= RERANK =================

def rerank_chunks(query, chunks, top_k=6):

    pairs = [[query, chunk] for chunk in chunks]

    scores = reranker.predict(pairs)

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    return [chunk for chunk, score in ranked[:top_k]]


# ================= FILE UPLOAD =================

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...), session_id: str = "default"):

    content = await file.read()
    extracted_text = ""
    filename = file.filename.lower()

    try:

        if filename.endswith(".pdf"):

            with pdfplumber.open(io.BytesIO(content)) as pdf:

                for page in pdf.pages:

                    text = page.extract_text()

                    if text:
                        extracted_text += text.replace("\n", " ").strip() + " "

        elif filename.endswith(".docx"):

            doc = Document(io.BytesIO(content))

            for para in doc.paragraphs:
                extracted_text += para.text + " "

        elif filename.endswith(".txt"):

            extracted_text = content.decode(errors="ignore")

        elif filename.endswith((".png", ".jpg", ".jpeg")):

            image = Image.open(io.BytesIO(content))
            extracted_text = pytesseract.image_to_string(image)

            if len(extracted_text.strip()) < 20:

                return {
                    "status": "no_text_found",
                    "message": "Image contains little readable text."
                }

        else:
            return {"status": "unsupported_file_type"}

        if not extracted_text.strip():
            return {"status": "no_text_found"}

        chunks = chunk_text(extracted_text)

        embeddings = embed_model.encode(chunks)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings).astype("float32"))

        index_file = os.path.join(UPLOAD_DIR, f"{session_id}.faiss")
        chunk_file = os.path.join(UPLOAD_DIR, f"{session_id}_chunks.npy")

        faiss.write_index(index, index_file)
        np.save(chunk_file, np.array(chunks, dtype=object))

        return {"status": "file_indexed"}

    except Exception as e:

        print("FILE ERROR:", e)
        return {"status": "error_processing_file"}


# ================= CHAT =================

@app.post("/chat")
def chat_endpoint(req: ChatRequest):

    if research_index is None:
        return {
            "response": "System is still loading. Please try again in a few seconds.",
            "suggestions": [],
            "sources": []
        }

    update_memory(req.session_id, "user", req.message)
    save_to_firebase(req.session_id, "user", req.message)

    general_words = ["hi", "hello", "hey", "thanks", "thank you"]

    if req.message.lower().strip() in general_words:

        response = llm.generate(req.message, temperature=0.3)

        update_memory(req.session_id, "assistant", response)
        save_to_firebase(req.session_id, "assistant", response)

        return {
            "response": response,
            "suggestions": generate_followups(response),
            "sources": []
        }

    retrieved_chunks = []
    paper_counter = {}

    search_queries = generate_search_queries(req.message)

    for query in search_queries:

        query_embedding = embed_model.encode([query])

        D, I = research_index.search(
            np.array(query_embedding).astype("float32"),
            k=5
        )

        for i in I[0]:

            if 0 <= i < len(research_chunks):

                chunk = research_chunks[i]

                if chunk not in retrieved_chunks:

                    retrieved_chunks.append(chunk)

                    src = research_metadata[i]

                    citation = f"{src.get('author','Unknown')} ({src.get('year','Unknown')}) - {src.get('title','Unknown')}"

                    paper_counter[citation] = paper_counter.get(citation, 0) + 1

        tokenized_query = query.split()

        bm25_scores = bm25.get_scores(tokenized_query)

        top_indices = np.argsort(bm25_scores)[-5:]

        for i in top_indices:

            chunk = research_chunks[i]

            if chunk not in retrieved_chunks:

                retrieved_chunks.append(chunk)

                src = research_metadata[i]

                citation = f"{src.get('author','Unknown')} ({src.get('year','Unknown')}) - {src.get('title','Unknown')}"

                paper_counter[citation] = paper_counter.get(citation, 0) + 1


    if len(retrieved_chunks) > 6:
        retrieved_chunks = rerank_chunks(req.message, retrieved_chunks)

    context = "\n\n".join(retrieved_chunks[:6])

    history = ""

    if req.session_id in conversation_memory:
        for m in conversation_memory[req.session_id]:
            history += f"{m['role']}: {m['content']}\n"

    prompt = f"""
You are K-GPT, a kombucha scientific research assistant.

Conversation History:
{history}

Use the provided research context to answer accurately.

Context:
{context}

Question:
{req.message}
"""

    answer = llm.generate(prompt, temperature=0.3)
    answer = verify_answer(req.message, context, answer)

    update_memory(req.session_id, "assistant", answer)
    save_to_firebase(req.session_id, "assistant", answer)

    suggestions = generate_followups(answer)

    sorted_papers = sorted(
        paper_counter.items(),
        key=lambda x: x[1],
        reverse=True
    )

    sources = []

    import urllib.parse

    for citation, count in sorted_papers[:4]:

        src = next(
            (x for x in research_metadata
             if f"{x.get('author','')} ({x.get('year','')}) - {x.get('title','')}" == citation),
            None
        )

        link = ""
        if src:
            link = src.get("url") or src.get("doi")

        if not link:
            query = urllib.parse.quote(citation)
            link = f"https://scholar.google.com/scholar?q={query}"

        sources.append({
            "citation": citation,
            "link": link
        })

    return {
        "response": answer,
        "suggestions": suggestions,
        "sources": sources
    }


# ================= CLEAR FILE =================

@app.post("/clear-file")
def clear_file(session_id: str):

    index_file = os.path.join(UPLOAD_DIR, f"{session_id}.faiss")
    chunk_file = os.path.join(UPLOAD_DIR, f"{session_id}_chunks.npy")

    if os.path.exists(index_file):
        os.remove(index_file)

    if os.path.exists(chunk_file):
        os.remove(chunk_file)

    return {"status": "file_cleared"}


# ================= FEEDBACK =================

@app.post("/feedback")
def feedback_endpoint(req: FeedbackRequest):

    save_feedback(req.query, req.response, req.feedback)

    if req.feedback == "positive":
        reply = "Thanks for the positive feedback! I'll continue improving responses."
    else:
        reply = "Thanks for the feedback. I'll try to improve my answers."

    return {"response": reply}