import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:

    def __init__(self):

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # 📂 Load FAISS data
        data_path = os.path.join(base_dir, "data")

        print("🔄 Loading FAISS index...")

        self.index = faiss.read_index(os.path.join(data_path, "kombucha_index.faiss"))
        self.chunks = np.load(os.path.join(data_path, "chunks.npy"), allow_pickle=True)
        self.metadata = np.load(os.path.join(data_path, "metadata.npy"), allow_pickle=True)

        # ✅ Embedding model (same as used during indexing)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print("✅ FAISS Retriever Ready")

        # 📂 Memory (same as your old system)
        self.memory_path = os.path.join(base_dir, "data", "conversation_memory")


    def _load_memory(self):
        memory_context = ""

        if not os.path.exists(self.memory_path):
            return memory_context

        files = os.listdir(self.memory_path)

        for file in files[-3:]:  # last 3 conversations
            try:
                with open(os.path.join(self.memory_path, file), "r", encoding="utf-8") as f:
                    memory_context += f.read() + "\n\n"
            except:
                continue

        return memory_context


    def retrieve(self, query: str, k: int = 5):

        # 🔍 Convert query → embedding
        query_embedding = self.model.encode([query])

        # 🔍 Search FAISS
        D, I = self.index.search(query_embedding, k)

        main_context = ""
        sources = []

        for idx in I[0]:
            chunk = self.chunks[idx]
            meta = self.metadata[idx]

            main_context += chunk + "\n\n"

            sources.append({
                "title": meta.get("title", "Unknown"),
                "author": meta.get("author", ""),
                "year": meta.get("year", "")
            })

        # 🧠 Load memory
        memory_context = self._load_memory()

        # 🧾 Final combined context
        combined_context = f"""
### Knowledge Context:
{main_context}

### Past Conversation Memory:
{memory_context}
"""

        return combined_context, sources