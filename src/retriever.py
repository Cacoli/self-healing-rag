# src/retriever.py
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from src.state import RAGState, RAGStatus
import os

load_dotenv()

# --- Load once at startup (free, runs locally) ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Connect to Pinecone ---
pc    = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))


def retrieve(state: RAGState) -> RAGState:
    """
    Node 1 — Retriever
    Embeds the current query and pulls the top 4 most
    relevant chunks from Pinecone.
    """
    print(f"\n🔍 [RETRIEVER] Query: '{state['current_query']}'")

    # Embed the query
    query_vector = embedder.encode(state["current_query"]).tolist()

    # Search Pinecone
    results = index.query(
        vector=query_vector,
        top_k=4,
        include_metadata=True
    )

    # Pull out just the text from metadata
    chunks = [
        match["metadata"]["text"]
        for match in results["matches"]
        if "text" in match["metadata"]
    ]

    print(f"✅ [RETRIEVER] Found {len(chunks)} chunks")

    return {
        **state,
        "retrieved_chunks": chunks,
        "status": RAGStatus.GENERATING
    }