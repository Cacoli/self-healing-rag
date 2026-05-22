# app.py
import streamlit as st
from dotenv import load_dotenv
from src.graph import rag_graph
from src.state import RAGStatus

load_dotenv()

st.set_page_config(
    page_title="Self-Healing RAG",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Self-Healing RAG")
st.caption("Powered by LangGraph · Groq · Pinecone")
st.markdown("Ask anything about **RAG, LLMs, embeddings, agents, and transformers**.")

query = st.text_input("Your question", placeholder="e.g. How does attention work in transformers?")

if st.button("Ask") and query.strip():
    with st.spinner("Thinking..."):
        initial_state = {
            "original_query":   query,
            "current_query":    query,
            "retrieved_chunks": [],
            "generated_answer": "",
            "critic_verdict":   "",
            "critic_reasoning": "",
            "retry_count":      0,
            "max_retries":      2,
            "status":           RAGStatus.RETRIEVING,
            "final_answer":     None,
        }
        result = rag_graph.invoke(initial_state)

    st.success("Answer")
    st.write(result["final_answer"])

    with st.expander("📊 Pipeline details"):
        st.write(f"**Critic verdict:** {result['critic_verdict']}")
        st.write(f"**Critic reasoning:** {result['critic_reasoning']}")
        st.write(f"**Retries:** {result['retry_count']}")

    with st.expander("📄 Retrieved chunks"):
        for i, chunk in enumerate(result["retrieved_chunks"]):
            st.markdown(f"**Chunk {i+1}:** {chunk}")