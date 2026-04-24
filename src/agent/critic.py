"""
src/agent/critic.py
====================
Agentic critic — evaluates recommendation quality and assigns confidence.

The critic's job is to look at the top-k results and ask:
"Do these songs actually satisfy what the user described?"

It runs in two modes:

  OFFLINE mode (no API key)
    Rule-based confidence scoring using three signals:
      1. Genre/mood coverage — do top results match the requested category?
      2. Score variance     — high variance signals weak matches
      3. Top-similarity     — did the RAG retriever find close matches?

  ONLINE mode (ANTHROPIC_API_KEY set)
    Calls claude-sonnet-4-20250514 to critique the results in natural language
    and return a structured confidence assessment.

The critic returns a CritiqueResult with:
  confidence   : float 0–1 (how good the recommendations are)
  note         : str  (plain-language explanation)
  should_rerank: bool (whether the planner should try again)
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CritiqueResult:
    """The output of one critic evaluation."""
    confidence: float       # 0.0 (terrible) to 1.0 (perfect)
    note: str               # plain-language critique
    should_rerank: bool     # True if planner should try a different mode

    def label(self) -> str:
        """Return a human-readable confidence label."""
        if self.confidence >= 0.80:
            return "HIGH"
        if self.confidence >= 0.55:
            return "MEDIUM"
        return "LOW"


# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------

RERANK_THRESHOLD = 0.55   # confidence below this triggers re-rank
HIGH_SIMILARITY = 0.40   # RAG top-match score considered "good"
LOW_SIMILARITY = 0.20   # RAG top-match score considered "poor"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def critique(
    query: str,
    parsed_profile: Dict,
    top_results: List[Tuple[Dict, float, str]],
    top_rag_similarity: float,
    attempt: int = 1,
) -> CritiqueResult:
    """
    Evaluate the quality of the current top-k recommendations.

    Tries the online (LLM) critic first; falls back to rule-based if
    no API key is set or the API call fails.

    Parameters
    ----------
    query              : Original user query string.
    parsed_profile     : Sanitised profile dict from guardrails.
    top_results        : List of (song_dict, score, explanation) tuples.
    top_rag_similarity : Highest cosine similarity from the retriever.
    attempt            : Which re-rank attempt this is (caps at 2).

    Returns
    -------
    CritiqueResult with confidence, note, and should_rerank flag.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if api_key and attempt == 1:
        try:
            return _online_critique(
                query, parsed_profile, top_results, top_rag_similarity
            )
        except Exception as e:
            # Fall through to offline if API call fails
            note_prefix = f"[API error: {type(e).__name__}] "
    else:
        note_prefix = ""

    result = _offline_critique(
        query, parsed_profile, top_results, top_rag_similarity
    )
    result.note = note_prefix + result.note
    return result


# ---------------------------------------------------------------------------
# Offline (rule-based) critic
# ---------------------------------------------------------------------------


def _offline_critique(
    query: str,
    parsed_profile: Dict,
    top_results: List[Tuple[Dict, float, str]],
    top_rag_similarity: float,
) -> CritiqueResult:
    """
    Rule-based confidence scoring using three observable signals.

    Signal 1: Genre/mood coverage (0–0.40 points)
      How many of the top-3 results match the requested genre or mood?

    Signal 2: Score spread (0–0.30 points)
      A tight cluster of high scores = good catalog fit.
      A big drop from #1 to #2 = genre dominance, poor diversity.

    Signal 3: RAG similarity (0–0.30 points)
      How well did the query match the catalog embedding?
    """
    if not top_results:
        return CritiqueResult(
            confidence=0.0,
            note="No results to evaluate.",
            should_rerank=False,
        )

    notes = []
    score = 0.0

    # ── Signal 1: genre/mood coverage ────────────────────────────────────────
    requested_genre = parsed_profile.get("genre", "").lower()
    requested_mood = parsed_profile.get("mood", "").lower()
    top3 = top_results[:3]
    matches = sum(
        1 for song, _, _ in top3
        if song.get("genre", "") == requested_genre
        or song.get("mood", "") == requested_mood
    )
    coverage_score = round((matches / len(top3)) * 0.40, 3)
    score += coverage_score

    if matches == len(top3):
        notes.append(f"all top-{len(top3)} results match genre or mood")
    elif matches > 0:
        notes.append(f"{matches}/{len(top3)} top results match genre or mood")
    else:
        notes.append("no top results match requested genre or mood")

    # ── Signal 2: score spread ────────────────────────────────────────────────
    scores = [s for _, s, _ in top_results]
    top_score = scores[0] if scores else 0.0
    second_score = scores[1] if len(scores) > 1 else top_score
    drop = top_score - second_score if top_score > 0 else 0.0
    # Small drop (< 2.0) = healthy spread; large drop (> 5.0) = one dominant match
    spread_score = max(0.0, 0.30 - (drop / 20.0))
    score += round(spread_score, 3)

    if drop > 4.0:
        notes.append(
            f"large score drop #{1}→#{2} ({drop:.1f} pts) suggests one-song match")
    else:
        notes.append(f"healthy score spread across top results")

    # ── Signal 3: RAG retrieval quality ──────────────────────────────────────
    if top_rag_similarity >= HIGH_SIMILARITY:
        rag_score = 0.30
        notes.append("strong semantic match in catalog")
    elif top_rag_similarity >= LOW_SIMILARITY:
        rag_score = 0.15
        notes.append("moderate semantic match in catalog")
    else:
        rag_score = 0.05
        notes.append(
            "weak semantic match — query may describe genre not in catalog")
    score += rag_score

    confidence = round(min(score, 1.0), 4)
    full_note = "; ".join(notes) + f". [offline mode]"

    return CritiqueResult(
        confidence=confidence,
        note=full_note,
        should_rerank=confidence < RERANK_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# Online (LLM) critic
# ---------------------------------------------------------------------------


def _online_critique(
    query: str,
    parsed_profile: Dict,
    top_results: List[Tuple[Dict, float, str]],
    top_rag_similarity: float,
) -> CritiqueResult:
    """
    LLM-based critique using claude-sonnet-4-20250514.

    Sends the top results and user intent to the model and asks for
    a structured JSON assessment: confidence score + plain note.
    """
    import urllib.request

    results_summary = "\n".join(
        f"  #{i+1} {song['title']} — {song['genre']}/{song['mood']} "
        f"energy={song['energy']:.2f} score={score:.2f}"
        for i, (song, score, _) in enumerate(top_results[:5])
    )

    prompt = f"""You are evaluating a music recommendation system.

User request: "{query}"
Parsed intent: genre={parsed_profile.get('genre')}, mood={parsed_profile.get('mood')}, energy={parsed_profile.get('energy')}, tags={parsed_profile.get('preferred_mood_tags')}

Top recommendations:
{results_summary}

Rate the quality of these recommendations on a scale of 0.0 to 1.0.
Consider: do the songs match the user's described mood and energy? Is there variety?

Respond ONLY with valid JSON in this exact format:
{{"confidence": 0.85, "note": "Brief plain-language critique in one sentence.", "should_rerank": false}}"""

    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 200,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    text = data["content"][0]["text"].strip()
    # Strip markdown fences if model adds them
    text = text.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(text)

    return CritiqueResult(
        confidence=float(parsed.get("confidence", 0.5)),
        note=str(parsed.get("note", "LLM critique unavailable.")),
        should_rerank=bool(parsed.get("should_rerank", False)),
    )
