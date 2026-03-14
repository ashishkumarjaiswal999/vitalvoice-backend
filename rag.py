"""
rag.py — RAG pipeline for VitalVoice.
Loads medical documents into ChromaDB and retrieves relevant context.
Uses ChromaDB's default embedding function to avoid API version issues.
"""

import chromadb
from chromadb.utils import embedding_functions
from medical_docs import MEDICAL_DOCUMENTS
import os

# ── ChromaDB setup ─────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "vitalvoice_medical"

def get_embedding_function():
    """
    Use ChromaDB's built-in sentence transformer embeddings.
    This runs locally — no API calls needed for embeddings.
    """
    return embedding_functions.DefaultEmbeddingFunction()

def initialize_vector_db():
    """Initialize ChromaDB and load medical documents if not already loaded."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = get_embedding_function()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    if collection.count() == 0:
        print("Loading medical documents into vector database...")
        ids        = [doc["id"] for doc in MEDICAL_DOCUMENTS]
        documents  = [doc["content"] for doc in MEDICAL_DOCUMENTS]
        metadatas  = [{"category": doc["category"], "id": doc["id"]} for doc in MEDICAL_DOCUMENTS]

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Loaded {len(MEDICAL_DOCUMENTS)} medical documents into ChromaDB.")
    else:
        print(f"Vector DB already loaded with {collection.count()} documents.")

    return collection

def retrieve_medical_context(query: str, n_results: int = 3) -> str:
    """
    Retrieve relevant medical context from ChromaDB for a given query.
    Returns formatted string of relevant medical information.
    """
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        embedding_fn = get_embedding_function()

        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn
        )

        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count())
        )

        if not results["documents"] or not results["documents"][0]:
            return "No specific medical context found."

        context_parts = []
        for i, doc in enumerate(results["documents"][0]):
            category = results["metadatas"][0][i].get("category", "general")
            context_parts.append(
                f"[Medical Reference {i+1} - {category.upper()}]\n{doc}"
            )

        return "\n\n".join(context_parts)

    except Exception as e:
        print(f"RAG retrieval error: {e}")
        return "Medical knowledge base temporarily unavailable."
