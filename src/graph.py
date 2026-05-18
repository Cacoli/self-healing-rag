# src/graph.py
from langgraph.graph import StateGraph, END
from src.state import RAGState, RAGStatus
from src.retriever import retrieve
from src.generator import generate
from src.critic import critique
from src.reformulator import reformulate


def route_after_critic(state: RAGState) -> str:
    """
    Edge router — called after critic node.
    Decides which node runs next based on status.
    """
    if state["status"] == RAGStatus.DONE:
        return "done"
    elif state["status"] == RAGStatus.FALLBACK:
        return "fallback"
    elif state["status"] == RAGStatus.REFORMULATING:
        return "reformulate"
    return "fallback"  # safety default


def route_after_generator(state: RAGState) -> str:
    """
    Edge router — called after generator node.
    Skips critic if generator already flagged a fallback.
    """
    if state["status"] == RAGStatus.FALLBACK:
        return "fallback"
    return "critique"


def fallback_node(state: RAGState) -> RAGState:
    """
    Terminal node — graceful failure.
    Returns a honest response instead of a hallucination.
    """
    print(f"\n🛑 [FALLBACK] Returning graceful response")
    return {
        **state,
        "final_answer": state.get("final_answer") or
                        "I don't have enough reliable information to answer this accurately.",
        "status": RAGStatus.FALLBACK
    }


def build_graph() -> StateGraph:
    """
    Assembles all nodes and edges into the LangGraph.
    Returns a compiled, runnable graph.
    """
    graph = StateGraph(RAGState)

    # --- Register all nodes ---
    graph.add_node("retrieve",    retrieve)
    graph.add_node("generate",    generate)
    graph.add_node("critique",    critique)
    graph.add_node("reformulate", reformulate)
    graph.add_node("fallback",    fallback_node)

    # --- Entry point ---
    graph.set_entry_point("retrieve")

    # --- Edges ---
    # retrieve → generate (always)
    graph.add_edge("retrieve", "generate")

    # generate → critique OR fallback (conditional)
    graph.add_conditional_edges(
        "generate",
        route_after_generator,
        {
            "critique": "critique",
            "fallback": "fallback"
        }
    )

    # critique → done OR fallback OR reformulate (conditional)
    graph.add_conditional_edges(
        "critique",
        route_after_critic,
        {
            "done":       END,
            "fallback":   "fallback",
            "reformulate":"reformulate"
        }
    )

    # reformulate → retrieve (the retry loop)
    graph.add_edge("reformulate", "retrieve")

    # fallback → END
    graph.add_edge("fallback", END)

    return graph.compile()


# --- Build once, reuse everywhere ---
rag_graph = build_graph()