print("🚀 FILE STARTED EXECUTING")

import os
import threading

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import (
    save_feedback,
    generate_followups,
    process_query,
    initialize_core,
    initialize_rag
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= STARTUP =================
def startup():

    try:
        initialize_core()
        initialize_rag()

        print("🔥 SERVER READY")

    except Exception as e:
        print("❌ STARTUP ERROR:", e)


threading.Thread(
    target=startup,
    daemon=True
).start()


# ================= REQUEST MODELS =================
class ChatRequest(BaseModel):
    message: str
    session_id: str


class FeedbackRequest(BaseModel):
    query: str
    response: str
    feedback: str


# ================= ROOT =================
@app.get("/")
def home():
    return {"status": "OK"}


# ================= CHAT =================
@app.post("/chat")
def chat(req: ChatRequest):

    try:

        response, suggestions, sources = process_query(
            req.message
        )

        return {
            "response": response,
            "suggestions": suggestions,
            "sources": sources
        }

    except Exception as e:

        print("❌ CHAT ERROR:", e)

        return {
            "response": "⚠️ Internal server error.",
            "suggestions": [],
            "sources": []
        }


# ================= FEEDBACK =================
@app.post("/feedback")
def feedback(req: FeedbackRequest):

    try:

        save_feedback(
            req.query,
            req.response,
            req.feedback
        )

        return {"status": "saved"}

    except Exception as e:

        print("❌ FEEDBACK ERROR:", e)

        return {"status": "error"}


# ================= FILE UPLOAD =================
@app.post("/upload-file")
async def upload_file(
    file: UploadFile = File(...)
):

    try:

        content = await file.read()

        return {
            "status": "received",
            "filename": file.filename,
            "size": len(content)
        }

    except Exception as e:

        print("❌ FILE ERROR:", e)

        return {"status": "error"}


# ================= MAIN =================
if __name__ == "__main__":

    import uvicorn

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )