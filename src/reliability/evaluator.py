"""
src/reliability/evaluator.py
==============================
Reliability and evaluation experiments for VibeFinder 2.0.

Runs five query-type experiments, measures confidence scores and
re-rank rates, and prints a structured summary table.

Usage
-----
    python -m src.reliability.evaluator
"""

from typing import Dict, List, Tuple

from src.agent.planner import run_pipeline
from src.recommender import load_songs

CATALOG_PATH = "data/songs.csv"
RUNS_PER_TYPE = 5   # keep low for fast CI; raise to 10 for thorough eval


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENT_PROFILES: List[Tuple[str, object]] = [
    (
        "specific genre+mood",
        {"genre": "lofi", "mood": "chill", "energy": 0.4,
         "likes_acoustic": True, "preferred_decade": "",
         "preferred_mood_tags": ["focused", "mellow"]},
    ),
    (
        "natural language vibe",
        "something euphoric and driving for a workout",
    ),
    (
        "conflicting preferences",
        {"genre": "metal", "mood": "happy", "energy": 0.15,
         "likes_acoustic": True, "preferred_decade": "",
         "preferred_mood_tags": []},
    ),
    (
        "missing genre in catalog",
        {"genre": "bossa nova", "mood": "relaxed", "energy": 0.45,
         "likes_acoustic": True, "preferred_decade": "",
         "preferred_mood_tags": []},
    ),
    (
        "ambiguous short query",
        "good music",
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_experiments(songs=None) -> List[Dict]:
    """
    Run all experiments and return a list of result dicts.

    Each dict contains: query_type, avg_confidence, rerank_count, runs.
    """
    if songs is None:
        songs = load_songs(CATALOG_PATH)

    summary = []

    for label, profile in EXPERIMENT_PROFILES:
        confidences = []
        rerank_count = 0

        for _ in range(RUNS_PER_TYPE):
            try:
                result = run_pipeline(profile, k=5, songs=songs)
                confidences.append(result.critique.confidence)
                if result.rerank_triggered:
                    rerank_count += 1
            except ValueError:
                # Guardrail rejection counts as 0.0 confidence
                confidences.append(0.0)

        avg_conf = round(sum(confidences) / len(confidences),
                         3) if confidences else 0.0
        summary.append({
            "query_type": label,
            "avg_confidence": avg_conf,
            "rerank_count": rerank_count,
            "runs": RUNS_PER_TYPE,
        })

    return summary


def print_summary(summary: List[Dict]) -> None:
    """Print a formatted evaluation table."""
    print(f"\n{'='*62}")
    print("  VibeFinder 2.0 — Reliability Evaluation")
    print(f"{'='*62}")
    print(f"  {'Query type':<28}  {'Avg conf':>9}  {'Re-ranks':>9}  {'Runs':>5}")
    print(f"  {'-'*56}")

    for row in summary:
        bar_filled = round(row["avg_confidence"] * 10)
        bar = "█" * bar_filled + "░" * (10 - bar_filled)
        print(
            f"  {row['query_type']:<28}  "
            f"{row['avg_confidence']:>5.3f} {bar}  "
            f"{row['rerank_count']:>4}/{row['runs']:<4}"
        )

    print(f"  {'-'*56}")
    all_conf = [r["avg_confidence"] for r in summary]
    overall = round(sum(all_conf) / len(all_conf), 3)
    total_reranks = sum(r["rerank_count"] for r in summary)
    total_runs = sum(r["runs"] for r in summary)
    print(f"  {'Overall average':<28}  {overall:>5.3f}{'':>12}  {total_reranks:>4}/{total_runs:<4}")
    print(f"{'='*62}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("Running reliability experiments...")
    summary = run_experiments()
    print_summary(summary)
