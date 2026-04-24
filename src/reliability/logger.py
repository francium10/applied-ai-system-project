"""
src/reliability/logger.py
==========================
Structured JSON decision logger.

Every pipeline run writes one entry to logs/decisions.json (one JSON object
per line — newline-delimited JSON, easy to grep and parse).

The log is the audit trail: it records what the system received, what it
decided, and why. This makes the system's behaviour inspectable after the fact.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "decisions.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def log_decision(
    query: str,
    parsed_profile: Dict,
    rag_candidates: List[int],
    top_results: List[Dict],
    confidence: float,
    rerank_triggered: bool,
    critic_note: str,
    guardrail_errors: Optional[List[str]] = None,
    mode: str = "balanced",
) -> None:
    """
    Append one decision record to logs/decisions.json.

    Each record is a single JSON line (newline-delimited JSON format).
    The log directory is created automatically if it does not exist.

    Parameters
    ----------
    query            : The original user query string.
    parsed_profile   : The sanitised profile dict after guardrails.
    rag_candidates   : Song IDs returned by the RAG retriever.
    top_results      : List of {id, title, score} dicts for the top-k.
    confidence       : Overall confidence score (0.0–1.0).
    rerank_triggered : Whether the agentic critic triggered a re-rank.
    critic_note      : Plain-language note from the agentic critic.
    guardrail_errors : Any warnings or errors raised by guardrails.
    mode             : Scoring mode used (default "balanced").
    """
    LOG_DIR.mkdir(exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "mode": mode,
        "parsed_profile": parsed_profile,
        "guardrail_errors": guardrail_errors or [],
        "rag_candidates": rag_candidates,
        "top_results": top_results,
        "confidence": round(confidence, 4),
        "rerank_triggered": rerank_triggered,
        "critic_note": critic_note,
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def log_error(query: Any, errors: List[str]) -> None:
    """
    Append a guardrail rejection record to the log.

    Used when input fails validation and the pipeline is aborted.
    """
    LOG_DIR.mkdir(exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": str(query),
        "status": "rejected",
        "guardrail_errors": errors,
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_log(n: int = 10) -> List[Dict]:
    """
    Return the last n records from the decision log.

    Returns an empty list if the log file does not exist yet.
    """
    if not LOG_FILE.exists():
        return []

    lines = LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
    recent = lines[-n:] if len(lines) >= n else lines

    records = []
    for line in recent:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def clear_log() -> None:
    """Delete the log file. Useful for test teardown."""
    if LOG_FILE.exists():
        LOG_FILE.unlink()
