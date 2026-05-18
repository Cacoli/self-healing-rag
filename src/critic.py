# src/critic.py
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from src.state import RAGState, RAGStatus
import os

load_dotenv()

# --- Same Gemini model, acting as a separate critic agent ---
critic_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"), temperature=0.3)


CRITIC_PROMPT = """You are a strict fact-checking agent. Your job is to decide if an answer is genuinely grounded in the provided context or if it contains hallucinations.

CONTEXT (the only source of truth):
{context}

QUESTION that was asked:
{question}

ANSWER to evaluate:
{answer}

Your evaluation rules:
1. PASS if every key claim in the answer can be traced back to the context
2. FAIL if the answer introduces facts, names, numbers, or claims NOT present in the context
3. FAIL if the answer is vague, evasive, or refuses to answer when the context clearly contains the information
4. FAIL if the answer contradicts the context in any way

Respond in this EXACT format and nothing else:
VERDICT: PASS or FAIL
REASONING: one sentence explaining your decision
"""


def critique(state: RAGState) -> RAGState:
    """
    Node 3 — Critic Agent
    Evaluates whether the generated answer is grounded
    in the retrieved chunks or contains hallucinations.
    """
    print(f"\n🧐 [CRITIC] Evaluating answer...")

    context = "\n\n---\n\n".join(state["retrieved_chunks"])

    prompt = CRITIC_PROMPT.format(
        context=context,
        question=state["original_query"],  # always the ORIGINAL question
        answer=state["generated_answer"]
    )

    response = critic_llm.invoke(prompt)
    raw      = response.content.strip()

    # --- Parse the verdict ---
    verdict   = "FAIL"  # default to FAIL if parsing breaks
    reasoning = raw

    for line in raw.split("\n"):
        if line.startswith("VERDICT:"):
            verdict = "PASS" if "PASS" in line else "FAIL"
        if line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()

    print(f"{'✅' if verdict == 'PASS' else '❌'} [CRITIC] Verdict: {verdict}")
    print(f"   Reasoning: {reasoning}")

    # --- Route based on verdict ---
    if verdict == "PASS":
        return {
            **state,
            "critic_verdict":   verdict,
            "critic_reasoning": reasoning,
            "final_answer":     state["generated_answer"],
            "status":           RAGStatus.DONE
        }
    else:
        # Check if we've hit the retry limit
        if state["retry_count"] >= state["max_retries"]:
            print(f"⚠️  [CRITIC] Max retries ({state['max_retries']}) reached → Fallback")
            return {
                **state,
                "critic_verdict":   verdict,
                "critic_reasoning": reasoning,
                "final_answer":     "I don't have enough reliable information to answer this question accurately.",
                "status":           RAGStatus.FALLBACK
            }

        # Otherwise → reformulate and retry
        return {
            **state,
            "critic_verdict":   verdict,
            "critic_reasoning": reasoning,
            "status":           RAGStatus.REFORMULATING
        }