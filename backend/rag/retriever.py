import os
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex
)
from llama_index.embeddings.ollama import OllamaEmbedding


class Retriever:

    def __init__(self):

        Settings.embed_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434"
        )

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        storage_context = StorageContext.from_defaults(
            persist_dir=os.path.join(base_dir, "rag", "index")
        )

        self.main_index = load_index_from_storage(storage_context)

        self.memory_path = os.path.join(
            base_dir,
            "data",
            "conversation_memory"
        )

        self.memory_index = None
        self._load_memory_index()

    def _load_memory_index(self):

        if not os.path.exists(self.memory_path):
            return

        if not os.listdir(self.memory_path):
            return

        documents = SimpleDirectoryReader(
            self.memory_path
        ).load_data()

        if documents:
            self.memory_index = VectorStoreIndex.from_documents(
                documents
            )

    def retrieve(self, query: str, k: int):

        self._load_memory_index()

        # -------- Main Retrieval --------
        main_retriever = self.main_index.as_retriever(
            similarity_top_k=k
        )

        main_nodes = main_retriever.retrieve(query)

        main_context = "\n\n".join(
            [node.text for node in main_nodes]
        )

        # -------- Extract Sources --------
        sources = []

        for node in main_nodes[:3]:
            filename = node.metadata.get(
                "file_name",
                "Unknown Source"
            )

            snippet = node.text[:300].replace("\n", " ")

            sources.append({
                "file": filename,
                "snippet": snippet
            })

        # -------- Memory Retrieval --------
        memory_context = ""

        if self.memory_index is not None:

            memory_retriever = self.memory_index.as_retriever(
                similarity_top_k=2
            )

            memory_nodes = memory_retriever.retrieve(query)

            memory_context = "\n\n".join(
                [node.text for node in memory_nodes]
            )

        combined_context = f"""
### Knowledge Context:
{main_context}

### Past Conversation Memory:
{memory_context}
"""

        return combined_context, sources