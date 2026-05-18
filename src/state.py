# src/state.py
from typing import TypedDict, List, Optional
from enum import Enum

class RAGStatus(Enum):
    RETRIEVING     = "retrieving"
    GENERATING     = "generating"
    CRITIQUING     = "critiquing"
    REFORMULATING  = "reformulating"
    DONE           = "done"
    FALLBACK       = "fallback"

class RAGState(TypedDict):
    # --- Input ---
    original_query:     str           # never changes — the user's original question

    # --- Retrieval ---
    current_query:      str           # may be reformulated on retry
    retrieved_chunks:   List[str]     # raw text chunks from Pinecone

    # --- Generation ---
    generated_answer:   str           # Claude/Gemini's answer

    # --- Critic ---
    critic_verdict:     str           # "PASS" or "FAIL"
    critic_reasoning:   str           # why it passed or failed

    # --- Control flow ---
    retry_count:        int           # how many times we've retried
    max_retries:        int           # hard limit (we'll set to 2)
    status:             RAGStatus     # current node we're in

    # --- Final output ---
    final_answer:       Optional[str] # what gets returned to the user