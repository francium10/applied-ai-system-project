"""
Tests for the Music Recommender Simulation.
==========================================
Run with:  pytest

Tests cover:
- Sorting correctness: better-matching songs rank higher
- Explanation validity: non-empty, human-readable strings
- Score range: scores are always non-negative
- Edge cases: k larger than catalog, empty reasons
"""

from src.recommender import Song, UserProfile, Recommender, score_song


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def make_small_recommender() -> Recommender:
    """Two-song catalog: one perfect pop/happy match, one lofi/chill."""
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def make_pop_user() -> UserProfile:
    return UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )


# ---------------------------------------------------------------------------
# Required starter tests (must not be modified)
# ---------------------------------------------------------------------------


def test_recommend_returns_songs_sorted_by_score():
    """The pop/happy song must rank first for a pop/happy user."""
    user = make_pop_user()
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    """Explanation must be a non-blank string."""
    user = make_pop_user()
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


# ---------------------------------------------------------------------------
# Extended tests added by student
# ---------------------------------------------------------------------------


def test_score_is_non_negative():
    """Scores should never go below zero regardless of mismatch."""
    user_prefs = {
        "genre": "classical",   # not in catalog at all
        "mood": "sad",
        "energy": 0.1,
        "likes_acoustic": True,
    }
    song = {
        "genre": "rock",
        "mood": "intense",
        "energy": 0.95,
        "valence": 0.3,
        "danceability": 0.6,
        "acousticness": 0.05,
    }
    score, _ = score_song(user_prefs, song)
    assert score >= 0.0


def test_genre_match_increases_score():
    """A song whose genre matches should outscore an identical song that doesn't."""
    base_song = {
        "genre": "pop",
        "mood": "energetic",    # won't match
        "energy": 0.5,
        "valence": 0.7,
        "danceability": 0.6,
        "acousticness": 0.2,
    }
    different_genre_song = {**base_song, "genre": "country"}

    user_prefs = {"genre": "pop", "mood": "happy",
                  "energy": 0.5, "likes_acoustic": False}

    score_match, _ = score_song(user_prefs, base_song)
    score_no_match, _ = score_song(user_prefs, different_genre_song)

    assert score_match > score_no_match


def test_recommend_k_larger_than_catalog():
    """When k > catalog size, all available songs are returned."""
    user = make_pop_user()
    rec = make_small_recommender()
    results = rec.recommend(user, k=100)
    assert len(results) == len(rec.songs)


def test_energy_proximity_rewards_closeness():
    """A song with energy 0.8 should outscore energy 0.2 for a user targeting 0.8."""
    user_prefs = {"genre": "other", "mood": "other",
                  "energy": 0.8, "likes_acoustic": False}
    close_song = {"genre": "x", "mood": "x", "energy": 0.8,
                  "valence": 0.5, "danceability": 0.5, "acousticness": 0.1}
    far_song = {"genre": "x", "mood": "x", "energy": 0.2,
                "valence": 0.5, "danceability": 0.5, "acousticness": 0.1}

    score_close, _ = score_song(user_prefs, close_song)
    score_far,   _ = score_song(user_prefs, far_song)

    assert score_close > score_far
