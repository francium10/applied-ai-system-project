"""
src/agent/planner.py
=====================
End-to-end pipeline orchestrator for VibeFinder 2.0.

The planner wires all components together in sequence:

  1. Guardrails    — validate and sanitise input
  2. RAG retriever — find semantically similar candidate songs
  3. Scoring engine— rank candidates by weighted feature match
  4. Critic        — evaluate quality, assign confidence
  5. Re-rank loop  — if confidence is low, try a different scoring mode
  6. Logger        — write full decision trace to JSON

Usage
-----
  python -m src.agent.planner --query "euphoric morning run"
  python -m src.agent.planner --genre pop --mood happy --energy 0.8

Or import and call directly:
  from src.agent.planner import run_pipeline
  result = run_pipeline("something mellow for studying")
  print(result)
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.agent.critic import CritiqueResult, critique
from src.rag.embedder import embed_query
from src.rag.retriever import Retriever
from src.rag.vector_store import VectorStore
from src.recommender import (
    ScoringMode,
    load_songs,
    recommend_songs,
)
from src.reliability.guardrails import validate_query
from src.reliability.logger import log_decision, log_error

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RERANK_ATTEMPTS = 2
DEFAULT_K = 5
CATALOG_PATH = "data/songs.csv"

# Fallback mode sequence: if BALANCED produces low confidence, try MOOD_FIRST
RERANK_MODE_SEQUENCE = [
    ScoringMode.BALANCED,
    ScoringMode.MOOD_FIRST,
    ScoringMode.ENERGY_FOCUS,
]


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """The complete output of one pipeline run."""
    query: str
    # (song, score, explanation)
    recommendations: List[Tuple[Dict, float, str]]
    critique: CritiqueResult
    mode_used: ScoringMode
    rerank_triggered: bool
    guardrail_errors: List[str]

    def __str__(self) -> str:
        return _format_result(self)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pipeline(
    query_or_profile,
    k: int = DEFAULT_K,
    songs: Optional[List[Dict]] = None,
) -> PipelineResult:
    """
    Run the full VibeFinder 2.0 pipeline on a query or profile.

    Parameters
    ----------
    query_or_profile : str (natural language) or dict (structured profile)
    k                : number of recommendations to return
    songs            : pre-loaded song list (loads from CSV if None)

    Returns
    -------
    PipelineResult with recommendations, critique, and metadata.
    Raises SystemExit if the input is hard-rejected by guardrails.
    """
    # ── Step 1: Guardrails ────────────────────────────────────────────────────
    clean_profile, errors = validate_query(query_or_profile)

    # Hard failures (empty query, None fields) abort the pipeline
    hard_failures = [
        e for e in errors if "cannot be" in e or "not supported" in e]
    if hard_failures:
        log_error(query_or_profile, errors)
        raise ValueError(
            "[GUARDRAIL] Input rejected:\n" +
            "\n".join(f"  - {e}" for e in hard_failures)
        )

    # ── Step 2: Load catalog ──────────────────────────────────────────────────
    if songs is None:
        songs = load_songs(CATALOG_PATH)

    # ── Step 3: Build / use vector store ─────────────────────────────────────
    store = VectorStore()
    store.build(songs)
    retriever = Retriever(store)

    # Use the natural language query if provided, otherwise build one from profile
    query_text = clean_profile.get(
        "query") or _profile_to_query_text(clean_profile)
    top_similarity = retriever.top_similarity(query_text)

    # Retrieve candidate set (larger pool for scoring to work with)
    candidates = retriever.retrieve(query_text, k=min(len(songs), 15))

    # ── Steps 4–5: Score → Critique → Re-rank loop ───────────────────────────
    mode = RERANK_MODE_SEQUENCE[0]
    results = recommend_songs(clean_profile, candidates, k=k, mode=mode)
    critique_result = critique(
        query_text, clean_profile, results, top_similarity, attempt=1
    )

    rerank_triggered = False

    if critique_result.should_rerank:
        rerank_triggered = True
        for attempt, fallback_mode in enumerate(RERANK_MODE_SEQUENCE[1:], start=2):
            if attempt > MAX_RERANK_ATTEMPTS:
                break
            results = recommend_songs(
                clean_profile, candidates, k=k, mode=fallback_mode
            )
            critique_result = critique(
                query_text, clean_profile, results,
                top_similarity, attempt=attempt
            )
            mode = fallback_mode
            if not critique_result.should_rerank:
                break

    # ── Step 6: Log decision ──────────────────────────────────────────────────
    log_decision(
        query=query_text,
        parsed_profile=clean_profile,
        rag_candidates=[s.get("id", 0) for s in candidates],
        top_results=[
            {"id": s.get("id"), "title": s.get("title"), "score": round(sc, 2)}
            for s, sc, _ in results
        ],
        confidence=critique_result.confidence,
        rerank_triggered=rerank_triggered,
        critic_note=critique_result.note,
        guardrail_errors=errors,
        mode=mode.value,
    )

    return PipelineResult(
        query=query_text,
        recommendations=results,
        critique=critique_result,
        mode_used=mode,
        rerank_triggered=rerank_triggered,
        guardrail_errors=errors,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

BAR_FULL = 14
MAX_SCORE = 11.5


def _score_bar(score: float) -> str:
    filled = round((score / MAX_SCORE) * BAR_FULL)
    empty = BAR_FULL - filled
    return f"[{'X'*filled}{'.'*empty}] {score:5.2f}"


def _format_result(result: PipelineResult) -> str:
    lines = []
    w = "=" * 62

    lines.append(f"\n{w}")
    lines.append(f"  VibeFinder 2.0  |  mode: {result.mode_used.value}")
    if result.rerank_triggered:
        lines.append("  [Re-rank was triggered by the agentic critic]")
    lines.append(w)

    if result.guardrail_errors:
        lines.append("  Guardrail notes:")
        for e in result.guardrail_errors:
            lines.append(f"    ! {e}")
        lines.append("")

    for rank, (song, score, explanation) in enumerate(result.recommendations, 1):
        lines.append(f"\n  #{rank}  {song['title']} — {song['artist']}")
        lines.append(
            f"       {song['genre']} / {song['mood']} / "
            f"energy {song['energy']:.2f}"
        )
        lines.append(f"       Score : {_score_bar(score)}")
        for reason in explanation.split("; "):
            if reason:
                lines.append(f"               * {reason}")

    lines.append(f"\n{'-'*62}")
    conf = result.critique
    lines.append(
        f"  Confidence: {conf.label()} ({conf.confidence:.2f})  |  "
        f"Critic: {conf.note}"
    )
    lines.append(f"  Log written to logs/decisions.json")
    lines.append(f"{w}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _profile_to_query_text(profile: Dict) -> str:
    """Convert a structured profile to a short query string for embedding."""
    parts = [
        profile.get("genre", ""),
        profile.get("mood", ""),
    ] + profile.get("preferred_mood_tags", [])
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(
        description="VibeFinder 2.0 — AI-augmented music recommender"
    )
    parser.add_argument("--query", "-q", type=str, default="",
                        help="Natural language query")
    parser.add_argument("--genre", type=str, default="pop")
    parser.add_argument("--mood", type=str, default="happy")
    parser.add_argument("--energy", type=float, default=0.7)
    parser.add_argument("--acoustic", action="store_true")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.query:
        user_input = args.query
    else:
        user_input = {
            "genre": args.genre,
            "mood": args.mood,
            "energy": args.energy,
            "likes_acoustic": args.acoustic,
        }

    try:
        result = run_pipeline(user_input, k=args.k)
        print(result)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
