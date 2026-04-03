print("🚀 FILE STARTED EXECUTING")
import io
import os
import faiss
import numpy as np
import pdfplumber
import pytesseract
import threading
import time

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
    pass


# ================= MODELS =================

embed_model = None
reranker = None

UPLOAD_DIR = "uploaded_indexes"
os.makedirs(UPLOAD_DIR, exist_ok=True)

DATA_DIR = "data"

INDEX_PATH = os.path.join(DATA_DIR, "kombucha_index.faiss")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.npy")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.npy")

research_index = None
research_chunks = None
research_metadata = None
bm25 = None

# ================= 🔥 LAZY INITIALIZER =================

model_loaded = False

def initialize_system():
    global research_index, research_chunks, research_metadata, bm25
    global embed_model, reranker, model_loaded

    if model_loaded:
        return

    print("🚀 Initializing system...")

    try:
        download_files()
    except:
        pass

    research_index = faiss.read_index(INDEX_PATH)
    research_chunks = np.load(CHUNKS_PATH, allow_pickle=True).tolist()
    research_metadata = np.load(METADATA_PATH, allow_pickle=True).tolist()

    tokenized_corpus = [chunk.split() for chunk in research_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    model_loaded = True
    print("✅ System ready")


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
        return llm.generate(prompt, temperature=0.2)
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

    if embed_model is None:
        return {"status": "system_not_ready"}

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

        else:
            return {"status": "unsupported_file_type"}

        if not extracted_text.strip():
            return {"status": "no_text_found"}

        chunks = chunk_text(extracted_text)
        embeddings = embed_model.encode(chunks)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings).astype("float32"))

        faiss.write_index(index, os.path.join(UPLOAD_DIR, f"{session_id}.faiss"))
        np.save(os.path.join(UPLOAD_DIR, f"{session_id}_chunks.npy"), chunks)

        return {"status": "file_indexed"}

    except Exception as e:
        print("FILE ERROR:", e)
        return {"status": "error_processing_file"}


# ================= CHAT =================

@app.post("/chat")
def chat_endpoint(req: ChatRequest):

    initialize_system()  # 🔥 KEY FIX

    update_memory(req.session_id, "user", req.message)

    # (rest of your logic unchanged)

    # ================= MAIN =================

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 10000))

    print(f"🚀 Starting server on port {port}")

    uvicorn.run(app, host="0.0.0.0", port=port)