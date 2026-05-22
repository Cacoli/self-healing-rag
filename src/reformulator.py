# src/reformulator.py
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from src.state import RAGState, RAGStatus
import os

load_dotenv()

REFORMULATION_PROMPT = """You are a search query optimizer. A previous search query failed to retrieve useful information to answer a question.

ORIGINAL QUESTION from user:
{original_query}

PREVIOUS QUERY that failed:
{failed_query}

CRITIC'S REASON it failed:
{critic_reasoning}

Your job: Write a NEW search query that approaches the question differently.
Rules:
1. Use different keywords and phrasing than the failed query
2. Break the question into its core concept if it was too complex
3. Try synonyms or related terms
4. Keep it concise — 1 sentence max
5. Output ONLY the new query, nothing else. No explanation, no punctuation at the end.
"""


def reformulate(state: RAGState) -> RAGState:
    load_dotenv()
    reformulator_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7
    )

    print(f"\n🔄 [REFORMULATOR] Retry {state['retry_count'] + 1}/{state['max_retries']}")
    print(f"   Failed query: '{state['current_query']}'")

    prompt = REFORMULATION_PROMPT.format(
        original_query   = state["original_query"],
        failed_query     = state["current_query"],
        critic_reasoning = state["critic_reasoning"]
    )

    response  = reformulator_llm.invoke(prompt)
    new_query = response.content.strip()

    print(f"   New query:    '{new_query}'")

    return {
        **state,
        "current_query": new_query,
        "retry_count":   state["retry_count"] + 1,
        "status":        RAGStatus.RETRIEVING
    }