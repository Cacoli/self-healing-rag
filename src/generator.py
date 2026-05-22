# src/generator.py
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from src.state import RAGState, RAGStatus
import os

load_dotenv()

GENERATION_PROMPT = """You are a helpful assistant. Answer the user's question using ONLY the context provided below.
If the context doesn't contain enough information, say "I don't have enough information to answer this."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def generate(state: RAGState) -> RAGState:
    load_dotenv()
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )

    print(f"\n🤖 [GENERATOR] Generating answer...")

    context = "\n\n---\n\n".join(state["retrieved_chunks"])

    if not context.strip():
        print("⚠️  [GENERATOR] No context found, triggering fallback")
        return {
            **state,
            "generated_answer": "",
            "status": RAGStatus.FALLBACK
        }

    prompt = GENERATION_PROMPT.format(
        context=context,
        question=state["current_query"]
    )

    response = llm.invoke(prompt)
    answer   = response.content.strip()

    print(f"✅ [GENERATOR] Answer generated ({len(answer)} chars)")

    return {
        **state,
        "generated_answer": answer,
        "status": RAGStatus.CRITIQUING
    }