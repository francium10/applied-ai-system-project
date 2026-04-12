"""
Music Recommender Simulation - Core Module
==========================================
Content-based filtering recommender that scores songs against a user
taste profile using weighted feature matching.

Architecture
------------
  Song            dataclass — track with audio + extended features
  UserProfile     dataclass — listener's taste preferences
  Recommender     OOP class — required by tests/test_recommender.py
  ScoringMode     enum      — Challenge 2: swappable ranking strategies
  load_songs      function  — CSV → list of dicts
  score_song      function  — scores one song (default BALANCED weights)
  score_song_with_weights   — scores one song with arbitrary weight dict
  recommend_songs function  — top-k with optional mode + diversity penalty
"""

import csv
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Song:
    """
    Represents a song and its audio + contextual attributes.

    Core audio features (original)
    -------------------------------
    id, title, artist, genre, mood, energy, tempo_bpm,
    valence, danceability, acousticness

    Extended features (Challenge 1)
    --------------------------------
    popularity     : 0–100 chart popularity score.
    release_decade : Era string, e.g. "2020s", "1990s".
    mood_tags      : Pipe-separated fine-grained mood descriptors,
                     e.g. "euphoric|driving|dark".
    """

    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    # Extended fields — default to safe values so old test fixtures still work
    popularity: float = 50.0
    release_decade: str = "2020s"
    mood_tags: str = ""


@dataclass
class UserProfile:
    """
    Represents a listener's taste preferences.

    Core fields
    -----------
    favorite_genre, favorite_mood, target_energy, likes_acoustic

    Extended fields (Challenge 1)
    -----------------------------
    preferred_decade  : Era preference, e.g. "2020s". Empty = no preference.
    preferred_mood_tags : List of fine-grained moods, e.g. ["euphoric","dark"].
    """

    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    preferred_decade: str = ""
    preferred_mood_tags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Challenge 2: Scoring Modes (Strategy Pattern)
# ---------------------------------------------------------------------------


class ScoringMode(Enum):
    """
    Swappable scoring strategies. Each mode re-weights the same features
    so the caller can choose what matters most without changing logic.

    BALANCED     Default — genre 3×, mood 2×, energy 2×.
    MOOD_FIRST   Emotional vibe dominates. Good for playlist curation.
    ENERGY_FOCUS Energy proximity is the top signal. Good for workout/study.
    GENRE_FIRST  Genre is the only categorical signal that matters.
    """
    BALANCED = "balanced"
    MOOD_FIRST = "mood_first"
    ENERGY_FOCUS = "energy_focus"
    GENRE_FIRST = "genre_first"


# Weight presets for each scoring mode (Challenge 2)
MODE_WEIGHTS: Dict[ScoringMode, Dict[str, float]] = {
    ScoringMode.BALANCED: {
        "genre": 3.0, "mood": 2.0, "energy": 2.0,
        "acousticness": 1.0, "valence": 1.0, "danceability": 0.5,
        "popularity": 0.5, "decade": 0.5, "mood_tags": 1.0,
    },
    ScoringMode.MOOD_FIRST: {
        "genre": 1.5, "mood": 4.0, "energy": 1.5,
        "acousticness": 0.5, "valence": 1.5, "danceability": 0.5,
        "popularity": 0.5, "decade": 0.5, "mood_tags": 2.0,
    },
    ScoringMode.ENERGY_FOCUS: {
        "genre": 1.0, "mood": 1.0, "energy": 4.0,
        "acousticness": 1.0, "valence": 0.5, "danceability": 1.5,
        "popularity": 0.5, "decade": 0.5, "mood_tags": 0.5,
    },
    ScoringMode.GENRE_FIRST: {
        "genre": 5.0, "mood": 1.0, "energy": 1.0,
        "acousticness": 1.0, "valence": 0.5, "danceability": 0.5,
        "popularity": 0.5, "decade": 0.5, "mood_tags": 0.5,
    },
}

# Convenience alias — default weights used by score_song()
WEIGHTS = MODE_WEIGHTS[ScoringMode.BALANCED]
MAX_POSSIBLE_SCORE: float = sum(WEIGHTS.values())


# ---------------------------------------------------------------------------
# OOP interface — required by tests/test_recommender.py
# ---------------------------------------------------------------------------


class Recommender:
    """
    Content-based music recommender (OOP interface for tests).

    Usage
    -----
    >>> rec = Recommender(songs)
    >>> top = rec.recommend(user, k=5)
    >>> print(rec.explain_recommendation(user, top[0]))
    """

    def __init__(self, songs: List[Song]) -> None:
        """Initialise with a list of Song dataclass instances."""
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top-k songs ranked by weighted relevance score."""
        scored: List[Tuple[Song, float]] = []
        for song in self.songs:
            score, _ = score_song(_profile_to_dict(user), _song_to_dict(song))
            scored.append((song, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0].title))
        return [song for song, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a human-readable explanation of why this song was suggested."""
        _, reasons = score_song(_profile_to_dict(user), _song_to_dict(song))
        return "; ".join(reasons) if reasons else "No strong match signals found."


# ---------------------------------------------------------------------------
# Functional interface — load_songs
# ---------------------------------------------------------------------------


def load_songs(csv_path: str) -> List[Dict]:
    """
    Load songs from a CSV file into a list of plain dictionaries.

    Numeric columns (energy, tempo_bpm, valence, danceability, acousticness,
    popularity) are cast to float. The id column is cast to int. String columns
    (genre, mood, release_decade, mood_tags) are stripped and lowercased.
    Extended columns are optional — missing columns get safe default values
    so the function works with both the original and expanded CSV schemas.

    Parameters
    ----------
    csv_path : str
        Path to the songs CSV file.

    Returns
    -------
    List[Dict]
        List of song dicts ready for score_song() and recommend_songs().
    """
    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "id":             int(row["id"]),
                "title":          row["title"],
                "artist":         row["artist"],
                "genre":          row["genre"].strip().lower(),
                "mood":           row["mood"].strip().lower(),
                "energy":         float(row["energy"]),
                "tempo_bpm":      float(row["tempo_bpm"]),
                "valence":        float(row["valence"]),
                "danceability":   float(row["danceability"]),
                "acousticness":   float(row["acousticness"]),
                # Challenge 1: extended features (graceful fallback if absent)
                "popularity":     float(row.get("popularity", 50)),
                "release_decade": row.get("release_decade", "2020s").strip(),
                "mood_tags":      row.get("mood_tags", "").strip().lower(),
            })
    return songs


# ---------------------------------------------------------------------------
# Functional interface — score_song
# ---------------------------------------------------------------------------


def score_song(
    user_prefs: Dict,
    song: Dict,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, List[str]]:
    """
    Score a single song against the user's preference profile.

    Scoring Algorithm
    -----------------
    Each feature contributes a partial score between 0 and its weight:

      genre        binary match × weight["genre"]
      mood         binary match × weight["mood"]
      energy       (1 - |target - actual|) × weight["energy"]
                   Proximity scoring: rewards songs *close* to target.
      acousticness binary alignment × weight["acousticness"]
      valence      (1 - |0.5 - valence|) × weight["valence"]
                   Rewards emotionally expressive songs (very + or very -)
      danceability song.danceability × weight["danceability"]
      popularity   (song.popularity / 100) × weight["popularity"]      [Ch.1]
      decade       binary match × weight["decade"]                     [Ch.1]
      mood_tags    (overlap_count / user_tag_count) × weight["mood_tags"] [Ch.1]

    The scoring rule answers "how relevant is THIS song?" (a single float).
    The ranking rule (recommend_songs) answers "which songs are best?" by
    sorting all those floats.

    Parameters
    ----------
    user_prefs : dict
        Keys: genre, mood, energy, likes_acoustic, preferred_decade (opt),
        preferred_mood_tags (opt, list of str).
    song : dict
        Song attributes from load_songs().
    weights : dict, optional
        Custom weight dict; defaults to BALANCED weights.

    Returns
    -------
    (total_score, reasons)
        total_score : float — the raw weighted relevance score
        reasons     : List[str] — human-readable match signals with points
    """
    w = weights if weights is not None else WEIGHTS
    score: float = 0.0
    reasons: List[str] = []

    # ── Genre (binary categorical) ─────────────────────────────────────────
    # Highest weight — genre is the most reliable proxy of musical taste.
    if song.get("genre", "").lower() == user_prefs.get("genre", "").lower():
        score += w["genre"]
        reasons.append(f"genre match — {song['genre']} (+{w['genre']})")

    # ── Mood (binary categorical) ──────────────────────────────────────────
    # Sets the emotional context of the listening session.
    if song.get("mood", "").lower() == user_prefs.get("mood", "").lower():
        score += w["mood"]
        reasons.append(f"mood match — {song['mood']} (+{w['mood']})")

    # ── Energy proximity (continuous) ─────────────────────────────────────
    # Proximity scoring: contribution = weight × (1 − distance).
    # A perfect match earns the full weight; a total mismatch earns 0.
    # Unlike a binary check, this rewards "close enough" without hard cutoffs.
    energy_distance = abs(user_prefs.get(
        "energy", 0.5) - song.get("energy", 0.5))
    energy_contribution = round(w["energy"] * (1.0 - energy_distance), 2)
    score += energy_contribution
    if energy_distance <= 0.15:
        reasons.append(
            f"energy match — {song['energy']:.2f} vs target "
            f"{user_prefs.get('energy', 0.5):.2f} (+{energy_contribution})"
        )

    # ── Acousticness (boolean alignment) ──────────────────────────────────
    # Threshold at 0.6: above = acoustic, below = electronic.
    song_is_acoustic = song.get("acousticness", 0.0) >= 0.6
    if user_prefs.get("likes_acoustic", False) == song_is_acoustic:
        score += w["acousticness"]
        label = "acoustic" if song_is_acoustic else "electronic"
        reasons.append(
            f"{label} feel matches preference (+{w['acousticness']})")

    # ── Valence (emotional clarity) ───────────────────────────────────────
    # Rewards emotionally expressive songs — clearly cheerful OR clearly dark.
    # Songs near 0.5 (emotionally neutral) score lowest on this dimension.
    valence = song.get("valence", 0.5)
    valence_contribution = round(w["valence"] * (1.0 - abs(0.5 - valence)), 2)
    score += valence_contribution

    # ── Danceability (continuous supporting signal) ────────────────────────
    danceability_contribution = round(
        w["danceability"] * song.get("danceability", 0.0), 2)
    score += danceability_contribution

    # ── Challenge 1: Popularity (continuous) ──────────────────────────────
    # Normalised to 0–1. Mild bonus for chart-performing tracks.
    popularity_contribution = round(
        w.get("popularity", 0) * (song.get("popularity", 50) / 100), 2)
    score += popularity_contribution

    # ── Challenge 1: Release decade (binary) ──────────────────────────────
    # Rewards songs from the user's preferred era. Ignored if no preference set.
    preferred_decade = user_prefs.get("preferred_decade", "")
    if preferred_decade and song.get("release_decade", "") == preferred_decade:
        score += w.get("decade", 0)
        reasons.append(
            f"era match — {song['release_decade']} (+{w.get('decade', 0)})")

    # ── Challenge 1: Fine-grained mood tags (overlap ratio) ───────────────
    # e.g. user wants ["euphoric", "driving"] and song has "euphoric|driving|dark"
    # → overlap = 2, user_count = 2 → ratio = 1.0 → full tag weight earned.
    user_tags: List[str] = user_prefs.get("preferred_mood_tags", [])
    if user_tags:
        song_tags: Set[str] = set(song.get("mood_tags", "").split("|"))
        overlap = len(set(user_tags) & song_tags)
        tag_ratio = overlap / len(user_tags)
        tag_contribution = round(w.get("mood_tags", 0) * tag_ratio, 2)
        score += tag_contribution
        if overlap > 0:
            matched = ", ".join(set(user_tags) & song_tags)
            reasons.append(
                f"mood tag overlap — {matched} (+{tag_contribution})")

    return round(score, 4), reasons


def score_song_with_weights(
    user_prefs: Dict, song: Dict, mode: ScoringMode
) -> Tuple[float, List[str]]:
    """Score a song using the weight preset for the given ScoringMode."""
    return score_song(user_prefs, song, weights=MODE_WEIGHTS[mode])


# ---------------------------------------------------------------------------
# Functional interface — recommend_songs
# ---------------------------------------------------------------------------


def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    mode: ScoringMode = ScoringMode.BALANCED,
    apply_diversity_penalty: bool = False,
) -> List[Tuple[Dict, float, str]]:
    """
    Score all songs, rank them, and return the top-k recommendations.

    This is the Ranking Rule. It uses score_song() as a judge for every song,
    then sorts all floats descending to find the best matches.

    Why sorted() instead of .sort()
    --------------------------------
    .sort()   mutates the original list in place and returns None.
    sorted()  returns a NEW sorted list; the original is untouched.
    We use sorted() to preserve the `songs` list for reuse across profiles.

    Challenge 2 — Scoring Modes
    ----------------------------
    Pass mode=ScoringMode.MOOD_FIRST, ENERGY_FOCUS, or GENRE_FIRST to swap
    the weight preset without changing any other logic.

    Challenge 3 — Diversity Penalty
    --------------------------------
    When apply_diversity_penalty=True, a song's score is multiplied by 0.6
    if its artist already appears among the selected top results. This prevents
    the same artist from dominating the list even when they have multiple
    high-scoring tracks.

    Parameters
    ----------
    user_prefs              : dict with genre, mood, energy, likes_acoustic, etc.
    songs                   : list of song dicts from load_songs().
    k                       : number of recommendations to return.
    mode                    : ScoringMode enum value (default BALANCED).
    apply_diversity_penalty : whether to penalise repeated artists.

    Returns
    -------
    List of (song_dict, score, explanation) tuples, sorted by score descending.
    Length is min(k, len(songs)).
    """
    w = MODE_WEIGHTS[mode]
    scored: List[Tuple[Dict, float, str]] = []

    # Scoring loop — every song gets judged by the same rules
    for song in songs:
        total_score, reasons = score_song(user_prefs, song, weights=w)
        explanation = "; ".join(
            reasons) if reasons else "General catalog suggestion"
        scored.append((song, total_score, explanation))

    # sorted() preserves the original `songs` list; negate for descending order
    ranked = sorted(scored, key=lambda item: (-item[1], item[0]["title"]))

    if not apply_diversity_penalty:
        return ranked[:k]

    # Challenge 3: Diversity Penalty
    # Walk the sorted list; if a song's artist already appears in selected,
    # multiply its score by 0.6 (a 40% penalty). Re-sort after adjustment.
    PENALTY = 0.6
    selected: List[Tuple[Dict, float, str]] = []
    seen_artists: Set[str] = set()

    for song, score, explanation in ranked:
        artist = song["artist"].lower()
        if artist in seen_artists:
            adjusted_score = round(score * PENALTY, 4)
            explanation = f"[diversity penalty applied] {explanation}"
            selected.append((song, adjusted_score, explanation))
        else:
            selected.append((song, score, explanation))
            seen_artists.add(artist)

    # Re-sort after penalties are applied
    selected.sort(key=lambda item: (-item[1], item[0]["title"]))
    return selected[:k]


# ---------------------------------------------------------------------------
# Private conversion helpers
# ---------------------------------------------------------------------------


def _song_to_dict(song: Song) -> Dict:
    """Convert a Song dataclass instance to a plain dict for scoring."""
    return {
        "id": song.id, "title": song.title, "artist": song.artist,
        "genre": song.genre, "mood": song.mood, "energy": song.energy,
        "tempo_bpm": song.tempo_bpm, "valence": song.valence,
        "danceability": song.danceability, "acousticness": song.acousticness,
        "popularity": song.popularity, "release_decade": song.release_decade,
        "mood_tags": song.mood_tags,
    }


def _profile_to_dict(user: UserProfile) -> Dict:
    """Convert a UserProfile dataclass instance to a plain dict for scoring."""
    return {
        "genre": user.favorite_genre, "mood": user.favorite_mood,
        "energy": user.target_energy, "likes_acoustic": user.likes_acoustic,
        "preferred_decade": user.preferred_decade,
        "preferred_mood_tags": user.preferred_mood_tags,
    }
