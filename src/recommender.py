"""
Music Recommender Simulation - Core Module
==========================================
Implements a content-based filtering recommender that scores songs
against a user taste profile using weighted feature matching.

Architecture:
  - Song         : dataclass representing a track and its audio features
  - UserProfile  : dataclass representing a listener's taste preferences
  - Recommender  : OOP class used by the test suite
  - load_songs   : functional helper — reads CSV into a list of dicts
  - score_song   : functional helper — scores one song against user prefs
  - recommend_songs : functional helper — ranks and returns top-k songs
"""

import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Song:
    """
    Represents a song and its audio attributes.

    Attributes
    ----------
    id            : Unique song identifier.
    title         : Display title of the track.
    artist        : Artist or band name.
    genre         : Primary genre label (e.g. "pop", "lofi", "rock").
    mood          : Emotional mood tag (e.g. "happy", "chill", "intense").
    energy        : Float 0–1. How energetic / intense the track feels.
    tempo_bpm     : Beats per minute. Pace / drive of the track.
    valence       : Float 0–1. Musical positiveness (high = cheerful).
    danceability  : Float 0–1. How suitable the track is for dancing.
    acousticness  : Float 0–1. Acoustic vs electronic instrument ratio.
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


@dataclass
class UserProfile:
    """
    Represents a listener's taste preferences.

    Attributes
    ----------
    favorite_genre  : The genre the user enjoys most.
    favorite_mood   : The mood the user typically seeks.
    target_energy   : Desired energy level (0–1).
    likes_acoustic  : Whether the user prefers acoustic over electronic.
    """

    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


# ---------------------------------------------------------------------------
# Scoring weights — edit these to experiment with different priorities
# ---------------------------------------------------------------------------

WEIGHTS: Dict[str, float] = {
    "genre": 3.0,       # Strong categorical match — most influential signal
    "mood": 2.0,        # Second most important — sets the emotional tone
    "energy": 2.0,      # Continuous proximity score — how close the vibe is
    "acousticness": 1.0,  # Bonus for matching acoustic preference
    "valence": 1.0,     # Mild positiveness alignment
    "danceability": 0.5,  # Subtle supporting feature
}

MAX_POSSIBLE_SCORE: float = sum(WEIGHTS.values())  # Used for normalizing


# ---------------------------------------------------------------------------
# OOP interface — required by tests/test_recommender.py
# ---------------------------------------------------------------------------


class Recommender:
    """
    Content-based music recommender.

    Wraps the functional scoring logic in an OOP interface so tests
    and the Streamlit UI can work with typed Song / UserProfile objects.

    Usage
    -----
    >>> rec = Recommender(songs)
    >>> top = rec.recommend(user, k=5)
    >>> print(rec.explain_recommendation(user, top[0]))
    """

    def __init__(self, songs: List[Song]) -> None:
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """
        Return the top-k songs ranked by weighted relevance score.

        Each Song is converted to a dict, scored, then re-mapped back
        to a Song object so the public API stays strongly typed.
        """
        scored: List[Tuple[Song, float]] = []

        for song in self.songs:
            song_dict = _song_to_dict(song)
            user_dict = _profile_to_dict(user)
            score, _ = score_song(user_dict, song_dict)
            scored.append((song, score))

        # Sort descending by score; break ties alphabetically by title
        scored.sort(key=lambda pair: (-pair[1], pair[0].title))
        return [song for song, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """
        Return a human-readable explanation of why this song was suggested.
        """
        song_dict = _song_to_dict(song)
        user_dict = _profile_to_dict(user)
        _, reasons = score_song(user_dict, song_dict)
        return "; ".join(reasons) if reasons else "No strong match signals found."


# ---------------------------------------------------------------------------
# Functional interface — required by src/main.py
# ---------------------------------------------------------------------------


def load_songs(csv_path: str) -> List[Dict]:
    """
    Load songs from a CSV file into a list of plain dictionaries.

    Each row in the CSV becomes a dict with properly typed values:
    numeric columns are cast to int or float automatically.

    Parameters
    ----------
    csv_path : Path to the songs CSV file.

    Returns
    -------
    List of song dicts ready for score_song() and recommend_songs().
    """
    songs: List[Dict] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append(
                {
                    "id": int(row["id"]),
                    "title": row["title"],
                    "artist": row["artist"],
                    "genre": row["genre"].strip().lower(),
                    "mood": row["mood"].strip().lower(),
                    "energy": float(row["energy"]),
                    "tempo_bpm": float(row["tempo_bpm"]),
                    "valence": float(row["valence"]),
                    "danceability": float(row["danceability"]),
                    "acousticness": float(row["acousticness"]),
                }
            )

    return songs


def score_song(
    user_prefs: Dict, song: Dict
) -> Tuple[float, List[str]]:
    """
    Score a single song against the user's preference profile.

    Scoring Algorithm
    -----------------
    This is a weighted content-based score. Each feature contributes
    a partial score between 0.0 and its weight:

      genre       → full weight if genre matches, else 0
      mood        → full weight if mood matches, else 0
      energy      → weight × (1 - |user_target - song_energy|)
                    Rewards songs *close* to the user's desired energy.
      acousticness→ weight if preferences align (both acoustic or both not)
      valence     → weight × (1 - |0.5 - song_valence|)
                    Rewards emotionally clear songs (very positive or moody)
      danceability→ weight × song_danceability (mild supporting signal)

    All partial scores are summed and returned alongside a list of
    human-readable reason strings explaining the key contributions.

    Returns
    -------
    (total_score, reasons)
        total_score : float — the raw weighted score
        reasons     : list of str — plain-language match signals
    """
    score: float = 0.0
    reasons: List[str] = []

    # --- Genre match (categorical) ----------------------------------------
    # Binary: either the genre matches exactly or it doesn't.
    # Weighted highest because genre is the strongest signal of musical taste.
    if song.get("genre", "").lower() == user_prefs.get("genre", "").lower():
        score += WEIGHTS["genre"]
        reasons.append(f"genre match — {song['genre']} (+{WEIGHTS['genre']})")

    # --- Mood match (categorical) -----------------------------------------
    # Binary: mood either aligns or it doesn't. Sets the emotional context.
    if song.get("mood", "").lower() == user_prefs.get("mood", "").lower():
        score += WEIGHTS["mood"]
        reasons.append(f"mood match — {song['mood']} (+{WEIGHTS['mood']})")

    # --- Energy proximity (continuous) ------------------------------------
    # Uses proximity scoring instead of a binary match:
    #   contribution = weight × (1 − distance)
    # A perfect match (distance=0) earns the full weight.
    # A total mismatch (distance=1) earns 0. No penalty below 0.
    # This is why we need a Scoring Rule separate from a binary check —
    # it rewards "close enough" without hard cutoffs.
    energy_distance = abs(user_prefs.get(
        "energy", 0.5) - song.get("energy", 0.5))
    energy_contribution = round(WEIGHTS["energy"] * (1.0 - energy_distance), 2)
    score += energy_contribution
    if energy_distance <= 0.15:
        reasons.append(
            f"energy match — {song['energy']:.2f} vs target "
            f"{user_prefs.get('energy', 0.5):.2f} (+{energy_contribution})"
        )

    # --- Acousticness preference (boolean alignment) ----------------------
    # Threshold at 0.6: songs above = acoustic, below = electronic.
    # Adds weight when the song's character aligns with the user's preference.
    song_is_acoustic = song.get("acousticness", 0.0) >= 0.6
    if user_prefs.get("likes_acoustic", False) == song_is_acoustic:
        score += WEIGHTS["acousticness"]
        label = "acoustic" if song_is_acoustic else "electronic"
        reasons.append(
            f"{label} feel matches preference (+{WEIGHTS['acousticness']})")

    # --- Valence (emotional clarity) --------------------------------------
    # Rewards songs that are emotionally expressive — either clearly cheerful
    # (valence → 1.0) or clearly melancholic (valence → 0.0).
    # Songs in the middle (valence ≈ 0.5) score lower on this dimension.
    valence = song.get("valence", 0.5)
    valence_contribution = round(
        WEIGHTS["valence"] * (1.0 - abs(0.5 - valence)), 2)
    score += valence_contribution

    # --- Danceability (mild supporting signal) ----------------------------
    # Proportional: a more danceable track gets a slightly higher score.
    # Low weight (0.5) so it never overrides the primary signals.
    danceability_contribution = round(
        WEIGHTS["danceability"] * song.get("danceability", 0.0), 2
    )
    score += danceability_contribution

    return round(score, 4), reasons


def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
) -> List[Tuple[Dict, float, str]]:
    """
    Score all songs, rank them, and return the top-k recommendations.

    This function is the Ranking Rule — it uses score_song() as a judge
    for every song in the catalog, then orders the results from best to worst.

    Why sorted() instead of .sort()
    --------------------------------
    list.sort()  — mutates the list IN PLACE, returns None.
                   Use when you want to permanently reorder the original list
                   and don't need to keep the unsorted version.

    sorted()     — returns a NEW sorted list, original is untouched.
                   Use when you want to preserve the original order, or when
                   sorting a generator / other iterable (not just lists).

    We use sorted() here because:
      1. We want to keep the original `songs` list unchanged for reusability.
      2. We're building `scored` as we go — sorted() on the final list is
         cleaner than building-then-sorting in place.

    Parameters
    ----------
    user_prefs : Dict with keys: genre, mood, energy, likes_acoustic.
    songs      : List of song dicts from load_songs().
    k          : Number of recommendations to return.

    Returns
    -------
    List of (song_dict, score, explanation_string) tuples, sorted by
    score descending. Length is min(k, len(songs)).
    """
    scored: List[Tuple[Dict, float, str]] = []

    # The Scoring Loop — every song gets judged by the same rules
    for song in songs:
        total_score, reasons = score_song(user_prefs, song)
        explanation = "; ".join(
            reasons) if reasons else "General catalog suggestion"
        scored.append((song, total_score, explanation))

    # sorted() returns a new list (original `songs` stays intact).
    # Key: negate score for descending order, use title as tiebreaker.
    ranked = sorted(scored, key=lambda item: (-item[1], item[0]["title"]))
    return ranked[:k]


# ---------------------------------------------------------------------------
# Private conversion helpers...
# ---------------------------------------------------------------------------


def _song_to_dict(song: Song) -> Dict:
    """Convert a Song dataclass instance to a plain dict for scoring."""
    return {
        "id": song.id,
        "title": song.title,
        "artist": song.artist,
        "genre": song.genre,
        "mood": song.mood,
        "energy": song.energy,
        "tempo_bpm": song.tempo_bpm,
        "valence": song.valence,
        "danceability": song.danceability,
        "acousticness": song.acousticness,
    }


def _profile_to_dict(user: UserProfile) -> Dict:
    """Convert a UserProfile dataclass instance to a plain dict for scoring."""
    return {
        "genre": user.favorite_genre,
        "mood": user.favorite_mood,
        "energy": user.target_energy,
        "likes_acoustic": user.likes_acoustic,
    }
