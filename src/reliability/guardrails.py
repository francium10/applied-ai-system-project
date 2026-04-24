"""
src/reliability/guardrails.py
==============================
Input validation and sanitisation layer.

Every request passes through validate_query() before reaching any AI component.
Bad inputs are rejected early with a structured error dict rather than silently
producing wrong answers deep in the pipeline.

Design principle: fail loudly at the boundary, not quietly in the middle.
"""

from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_GENRES = {
    "pop", "lofi", "rock", "indie pop", "ambient", "jazz", "synthwave",
    "electronic", "r&b", "classical", "funk", "folk", "gospel",
    "metal", "latin", "dream pop",
}

VALID_MOODS = {
    "happy", "chill", "intense", "relaxed", "moody", "focused",
    "melancholic", "nostalgic", "uplifting", "angry",
}

FLOAT_FIELDS = {"energy", "valence",
                "danceability", "acousticness", "target_energy"}
MAX_QUERY_LENGTH = 500
MIN_QUERY_LENGTH = 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_query(raw: Any) -> Tuple[Dict, List[str]]:
    """
    Validate and sanitise a raw input before it enters the pipeline.

    Accepts either:
      - a plain string  (natural language query)
      - a dict          (structured taste profile)

    Returns
    -------
    (clean, errors)
        clean  : sanitised dict ready for downstream processing
        errors : list of error strings; empty list means input is valid

    The returned dict always has these keys with safe defaults:
        query, genre, mood, energy, likes_acoustic,
        preferred_decade, preferred_mood_tags
    """
    errors: List[str] = []
    clean: Dict = _default_profile()

    if isinstance(raw, str):
        clean, errors = _validate_string_query(raw, clean, errors)
    elif isinstance(raw, dict):
        clean, errors = _validate_dict_profile(raw, clean, errors)
    else:
        errors.append(
            f"input type '{type(raw).__name__}' not supported; "
            "expected str or dict"
        )

    return clean, errors


def is_valid(raw: Any) -> bool:
    """Return True if the input passes all guardrail checks."""
    _, errors = validate_query(raw)
    return len(errors) == 0


def sanitise_energy(value: Any) -> Tuple[float, str | None]:
    """
    Cast value to float and clamp to [0.0, 1.0].

    Returns (clamped_value, warning_message_or_None).
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.5, f"energy '{value}' could not be parsed; defaulted to 0.5"

    if f < 0.0:
        return 0.0, f"energy {f} below 0.0; clamped to 0.0"
    if f > 1.0:
        return 1.0, f"energy {f} above 1.0; clamped to 1.0"
    return f, None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _default_profile() -> Dict:
    """Return a safe default profile dict."""
    return {
        "query": "",
        "genre": "",
        "mood": "",
        "energy": 0.5,
        "likes_acoustic": False,
        "preferred_decade": "",
        "preferred_mood_tags": [],
    }


def _validate_string_query(
    raw: str, clean: Dict, errors: List[str]
) -> Tuple[Dict, List[str]]:
    """Validate a free-text natural language query."""
    stripped = raw.strip()

    if len(stripped) < MIN_QUERY_LENGTH:
        errors.append(
            f"query too short ({len(stripped)} chars); "
            f"minimum is {MIN_QUERY_LENGTH}"
        )
        return clean, errors

    if len(stripped) > MAX_QUERY_LENGTH:
        stripped = stripped[:MAX_QUERY_LENGTH]
        errors.append(
            f"query truncated to {MAX_QUERY_LENGTH} characters"
        )

    clean["query"] = stripped
    return clean, errors


def _validate_dict_profile(
    raw: Dict, clean: Dict, errors: List[str]
) -> Tuple[Dict, List[str]]:
    """Validate a structured taste-profile dict."""

    # ── genre ────────────────────────────────────────────────────────────────
    genre = raw.get("genre", "")
    if genre is None or str(genre).strip() == "":
        errors.append("genre: cannot be empty")
    else:
        genre_lower = str(genre).strip().lower()
        clean["genre"] = genre_lower
        if genre_lower not in VALID_GENRES:
            # Warn but do not reject — unknown genre means no catalog match,
            # which the pipeline handles gracefully via RAG fallback.
            errors.append(
                f"genre '{genre_lower}' not in catalog; "
                "RAG fallback will activate"
            )

    # ── mood ─────────────────────────────────────────────────────────────────
    mood = raw.get("mood", "")
    if mood is None or str(mood).strip() == "":
        errors.append("mood: cannot be None or empty")
    else:
        mood_lower = str(mood).strip().lower()
        clean["mood"] = mood_lower
        if mood_lower not in VALID_MOODS:
            errors.append(
                f"mood '{mood_lower}' not recognised; "
                "scoring will fall back to energy + tag signals"
            )

    # ── energy ───────────────────────────────────────────────────────────────
    if "energy" in raw:
        clamped, warning = sanitise_energy(raw["energy"])
        clean["energy"] = clamped
        if warning:
            errors.append(warning)

    # ── likes_acoustic ───────────────────────────────────────────────────────
    if "likes_acoustic" in raw:
        val = raw["likes_acoustic"]
        if not isinstance(val, bool):
            errors.append(
                f"likes_acoustic must be bool; got '{type(val).__name__}'; "
                "defaulted to False"
            )
        else:
            clean["likes_acoustic"] = val

    # ── preferred_decade ─────────────────────────────────────────────────────
    decade = raw.get("preferred_decade", "")
    if decade:
        clean["preferred_decade"] = str(decade).strip()

    # ── preferred_mood_tags ──────────────────────────────────────────────────
    tags = raw.get("preferred_mood_tags", [])
    if not isinstance(tags, list):
        errors.append(
            f"preferred_mood_tags must be a list; got '{type(tags).__name__}'; "
            "defaulted to []"
        )
    else:
        clean["preferred_mood_tags"] = [
            str(t).strip().lower() for t in tags if t]

    # ── query passthrough (optional in dict mode) ─────────────────────────────
    if "query" in raw:
        q = str(raw["query"]).strip()
        clean["query"] = q[:MAX_QUERY_LENGTH]

    return clean, errors
