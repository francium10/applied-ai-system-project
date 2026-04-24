"""
src/rag/embedder.py
====================
Converts songs and queries into comparable embedding vectors.

We use a lightweight TF-IDF-style bag-of-words approach over a song's
feature string. This requires no external model, no API key, and no
dependencies beyond the standard library — making it fully offline.

Why not use a real embedding model?
  A 20-song catalog doesn't justify the latency or cost of calling an
  embedding API. The feature string approach captures the same semantic
  signal for this dataset: genre, mood, and tags dominate the vector,
  exactly as they dominate the scoring weights.

The interface is identical to what a real embedding model would expose:
  embed_song(song_dict)  -> list[float]
  embed_query(text)      -> list[float]
  cosine_similarity(a,b) -> float

So upgrading to sentence-transformers or the Anthropic embeddings API
later requires only changing this file, not the retriever.
"""

import math
import re
from typing import Dict, List

# ---------------------------------------------------------------------------
# Vocabulary — the feature tokens we care about most
# ---------------------------------------------------------------------------

# Ordered list of all tokens that can appear in a song's feature string.
# The position of each token is its index in the embedding vector.
_VOCAB: List[str] = [
    # Genres
    "pop", "lofi", "rock", "indie", "ambient", "jazz", "synthwave",
    "electronic", "r&b", "classical", "funk", "folk", "gospel", "metal",
    "latin", "dream",
    # Moods
    "happy", "chill", "intense", "relaxed", "moody", "focused",
    "melancholic", "nostalgic", "uplifting", "angry",
    # Mood tags
    "euphoric", "bright", "upbeat", "mellow", "dreamy", "aggressive",
    "driving", "raw", "peaceful", "soft", "cosmic", "warm", "cinematic",
    "dark", "minimal", "calm", "smooth", "sensual", "romantic", "breezy",
    "soulful", "playful", "hypnotic", "tender", "sad", "acoustic",
    # Energy descriptors
    "high", "low", "medium", "energetic", "quiet", "loud", "fast", "slow",
    # Context words users might type
    "run", "workout", "study", "sleep", "party", "morning", "night",
    "drive", "highway", "sad", "cry", "dance", "focus", "relax",
]

VOCAB_SIZE = len(_VOCAB)
_TOKEN_INDEX = {token: i for i, token in enumerate(_VOCAB)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_song(song: Dict) -> List[float]:
    """
    Convert a song dict into a feature embedding vector.

    The feature string concatenates genre, mood, and mood_tags so that
    all three contribute to the embedding in proportion to their importance
    in the scoring weights.
    """
    feature_string = _song_to_feature_string(song)
    return _vectorise(feature_string)


def embed_query(text: str) -> List[float]:
    """
    Convert a natural language query into an embedding vector.

    Tokenises the query and maps tokens to vocabulary positions.
    Unknown words are ignored — only known feature tokens contribute.
    """
    return _vectorise(text.lower())


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two equal-length vectors.

    Returns a float in [-1.0, 1.0]; higher = more similar.
    Returns 0.0 if either vector is the zero vector.
    """
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _song_to_feature_string(song: Dict) -> str:
    """
    Build a single text string from a song's most important features.

    Genre and mood are included twice to up-weight them — mirroring the
    higher weights they carry in the scoring engine (3.0 and 2.0).
    """
    parts = [
        song.get("genre", ""),
        song.get("genre", ""),       # doubled — mirrors genre weight 3.0
        song.get("mood", ""),
        song.get("mood", ""),        # doubled — mirrors mood weight 2.0
        song.get("mood_tags", "").replace("|", " "),
    ]
    return " ".join(p for p in parts if p).lower()


def _vectorise(text: str) -> List[float]:
    """
    Convert text into a bag-of-words vector over _VOCAB.

    Each position holds the normalised term frequency of that token.
    """
    vector = [0.0] * VOCAB_SIZE
    tokens = re.findall(r"[a-z&]+", text.lower())

    if not tokens:
        return vector

    for token in tokens:
        if token in _TOKEN_INDEX:
            vector[_TOKEN_INDEX[token]] += 1.0

    # L2-normalise so cosine similarity is well-behaved
    magnitude = math.sqrt(sum(x * x for x in vector))
    if magnitude > 0:
        vector = [x / magnitude for x in vector]

    return vector
