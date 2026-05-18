# main.py
from dotenv import load_dotenv
from src.graph import rag_graph
from src.state import RAGStatus

load_dotenv()


def ask(question: str) -> None:
    print("\n" + "="*60)
    print(f"❓ QUESTION: {question}")
    print("="*60)

    # --- Initial state ---
    initial_state = {
        "original_query":   question,
        "current_query":    question,
        "retrieved_chunks": [],
        "generated_answer": "",
        "critic_verdict":   "",
        "critic_reasoning": "",
        "retry_count":      0,
        "max_retries":      2,
        "status":           RAGStatus.RETRIEVING,
        "final_answer":     None,
    }

    # --- Run the graph ---
    final_state = rag_graph.invoke(initial_state)

    # --- Print result ---
    print("\n" + "="*60)
    print(f"✅ FINAL ANSWER:")
    print(final_state["final_answer"])
    print(f"\n📊 Stats:")
    print(f"   Retries:        {final_state['retry_count']}")
    print(f"   Critic verdict: {final_state['critic_verdict']}")
    print(f"   Critic reason:  {final_state['critic_reasoning']}")
    print("="*60)
if __name__ == "__main__":
    ask("What is the difference between fine-tuning and RAG?")
    ask("How does the attention mechanism work in transformers?")
    ask("What is RLHF and how does it align LLMs with human preferences?")
    ask("What is the difference between FAISS and Pinecone?")
    ask("What is the capital of France?")  # should still fallback