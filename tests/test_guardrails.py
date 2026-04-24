"""
tests/test_guardrails.py
=========================
Full test suite for the input validation and sanitisation layer.

Coverage map:
  validate_query() — string path, dict path, unsupported types
  is_valid()       — convenience boolean wrapper
  sanitise_energy()— clamp, parse errors, boundary values
  Field validation — genre, mood, energy, likes_acoustic, tags, decade
  Default profile  — safe defaults always populated
  Query truncation — MAX_QUERY_LENGTH enforcement

Run with:  pytest tests/test_guardrails.py -v
"""

import pytest
from src.reliability.guardrails import (
    validate_query,
    is_valid,
    sanitise_energy,
    VALID_GENRES,
    VALID_MOODS,
    MAX_QUERY_LENGTH,
    MIN_QUERY_LENGTH,
)


# ---------------------------------------------------------------------------
# 1. validate_query — string path
# ---------------------------------------------------------------------------

class TestStringQuery:
    def test_valid_string_passes(self):
        clean, errors = validate_query("euphoric morning run")
        assert clean["query"] == "euphoric morning run"
        assert errors == []

    def test_empty_string_rejected(self):
        _, errors = validate_query("")
        assert any("too short" in e for e in errors)

    def test_single_char_rejected(self):
        _, errors = validate_query("a")
        assert any("too short" in e for e in errors)

    def test_two_char_passes(self):
        clean, errors = validate_query("ok")
        assert any("too short" not in e for e in errors) or errors == []
        assert clean["query"] == "ok"

    def test_long_string_truncated(self):
        long = "a" * (MAX_QUERY_LENGTH + 50)
        clean, errors = validate_query(long)
        assert len(clean["query"]) <= MAX_QUERY_LENGTH
        assert any("truncated" in e for e in errors)

    def test_string_at_max_length_not_truncated(self):
        exact = "a" * MAX_QUERY_LENGTH
        clean, errors = validate_query(exact)
        truncation_errors = [e for e in errors if "truncated" in e]
        assert truncation_errors == []

    def test_whitespace_stripped(self):
        clean, _ = validate_query("  chill lofi  ")
        assert clean["query"] == "chill lofi"

    def test_default_profile_keys_always_present(self):
        clean, _ = validate_query("some query")
        for key in ["query", "genre", "mood", "energy",
                    "likes_acoustic", "preferred_decade", "preferred_mood_tags"]:
            assert key in clean


# ---------------------------------------------------------------------------
# 2. validate_query — dict path
# ---------------------------------------------------------------------------

class TestDictProfile:
    def test_valid_profile_passes(self):
        profile = {"genre": "pop", "mood": "happy", "energy": 0.8,
                   "likes_acoustic": False, "preferred_mood_tags": []}
        clean, errors = validate_query(profile)
        hard = [e for e in errors if "cannot be" in e]
        assert hard == []
        assert clean["genre"] == "pop"
        assert clean["mood"] == "happy"

    def test_empty_genre_hard_error(self):
        _, errors = validate_query({"genre": "", "mood": "happy", "energy": 0.5})
        assert any("genre" in e and "cannot be" in e for e in errors)

    def test_none_genre_hard_error(self):
        _, errors = validate_query({"genre": None, "mood": "happy", "energy": 0.5})
        assert any("genre" in e for e in errors)

    def test_none_mood_hard_error(self):
        _, errors = validate_query({"genre": "pop", "mood": None, "energy": 0.5})
        assert any("mood" in e and "cannot be" in e for e in errors)

    def test_empty_mood_hard_error(self):
        _, errors = validate_query({"genre": "pop", "mood": "", "energy": 0.5})
        assert any("mood" in e for e in errors)

    def test_energy_clamped_above_one(self):
        clean, errors = validate_query({"genre": "pop", "mood": "happy", "energy": 1.8})
        assert clean["energy"] == 1.0
        assert any("clamped" in e for e in errors)

    def test_energy_clamped_below_zero(self):
        clean, errors = validate_query({"genre": "pop", "mood": "happy", "energy": -0.5})
        assert clean["energy"] == 0.0
        assert any("clamped" in e for e in errors)

    def test_energy_in_range_no_clamp(self):
        clean, errors = validate_query({"genre": "pop", "mood": "happy", "energy": 0.7})
        assert clean["energy"] == 0.7
        assert not any("clamped" in e for e in errors)

    def test_likes_acoustic_bool_accepted(self):
        clean, errors = validate_query({"genre": "folk", "mood": "relaxed",
                                        "energy": 0.4, "likes_acoustic": True})
        hard = [e for e in errors if "cannot be" in e]
        assert hard == []
        assert clean["likes_acoustic"] is True

    def test_likes_acoustic_non_bool_soft_error(self):
        _, errors = validate_query({"genre": "pop", "mood": "happy",
                                    "energy": 0.5, "likes_acoustic": "yes"})
        assert any("likes_acoustic" in e for e in errors)

    def test_unknown_genre_soft_warning_not_hard(self):
        _, errors = validate_query({"genre": "bossa nova", "mood": "relaxed", "energy": 0.5})
        hard = [e for e in errors if "cannot be" in e]
        soft = [e for e in errors if "not in catalog" in e or "RAG fallback" in e]
        assert hard == []
        assert soft  # should have a soft warning

    def test_unknown_mood_soft_warning(self):
        _, errors = validate_query({"genre": "pop", "mood": "ecstatic", "energy": 0.5})
        assert any("not recognised" in e for e in errors)

    def test_preferred_decade_stored(self):
        clean, _ = validate_query({"genre": "pop", "mood": "happy",
                                   "energy": 0.7, "preferred_decade": "1990s"})
        assert clean["preferred_decade"] == "1990s"

    def test_preferred_mood_tags_stored(self):
        clean, _ = validate_query({"genre": "pop", "mood": "happy", "energy": 0.7,
                                   "preferred_mood_tags": ["euphoric", "bright"]})
        assert "euphoric" in clean["preferred_mood_tags"]
        assert "bright"   in clean["preferred_mood_tags"]

    def test_preferred_mood_tags_non_list_soft_error(self):
        _, errors = validate_query({"genre": "pop", "mood": "happy", "energy": 0.7,
                                    "preferred_mood_tags": "euphoric"})
        assert any("preferred_mood_tags" in e for e in errors)

    def test_tags_lowercased_and_stripped(self):
        clean, _ = validate_query({"genre": "pop", "mood": "happy", "energy": 0.7,
                                   "preferred_mood_tags": ["  EUPHORIC  ", "Bright"]})
        assert "euphoric" in clean["preferred_mood_tags"]
        assert "bright"   in clean["preferred_mood_tags"]

    def test_default_profile_keys_always_present(self):
        clean, _ = validate_query({"genre": "pop", "mood": "happy", "energy": 0.5})
        for key in ["query","genre","mood","energy",
                    "likes_acoustic","preferred_decade","preferred_mood_tags"]:
            assert key in clean

    def test_genre_normalised_to_lowercase(self):
        clean, _ = validate_query({"genre": "POP", "mood": "happy", "energy": 0.5})
        assert clean["genre"] == "pop"


# ---------------------------------------------------------------------------
# 3. validate_query — unsupported types
# ---------------------------------------------------------------------------

class TestUnsupportedTypes:
    def test_integer_rejected(self):
        _, errors = validate_query(42)
        assert any("not supported" in e for e in errors)

    def test_list_rejected(self):
        _, errors = validate_query(["pop", "happy"])
        assert any("not supported" in e for e in errors)

    def test_none_rejected(self):
        _, errors = validate_query(None)
        assert any("not supported" in e for e in errors)

    def test_float_rejected(self):
        _, errors = validate_query(3.14)
        assert any("not supported" in e for e in errors)


# ---------------------------------------------------------------------------
# 4. is_valid convenience wrapper
# ---------------------------------------------------------------------------

class TestIsValid:
    def test_valid_string_returns_true(self):
        assert is_valid("euphoric morning run") is True

    def test_empty_string_returns_false(self):
        assert is_valid("") is False

    def test_valid_dict_returns_true(self):
        assert is_valid({"genre": "pop", "mood": "happy", "energy": 0.7}) is True

    def test_dict_with_empty_genre_returns_false(self):
        assert is_valid({"genre": "", "mood": "happy", "energy": 0.7}) is False

    def test_unsupported_type_returns_false(self):
        assert is_valid(99) is False


# ---------------------------------------------------------------------------
# 5. sanitise_energy standalone
# ---------------------------------------------------------------------------

class TestSanitiseEnergy:
    def test_valid_float_unchanged(self):
        val, warning = sanitise_energy(0.75)
        assert val == 0.75
        assert warning is None

    def test_zero_valid(self):
        val, warning = sanitise_energy(0.0)
        assert val == 0.0
        assert warning is None

    def test_one_valid(self):
        val, warning = sanitise_energy(1.0)
        assert val == 1.0
        assert warning is None

    def test_above_one_clamped(self):
        val, warning = sanitise_energy(1.5)
        assert val == 1.0
        assert warning is not None and "clamped" in warning

    def test_below_zero_clamped(self):
        val, warning = sanitise_energy(-0.3)
        assert val == 0.0
        assert warning is not None

    def test_string_number_parsed(self):
        val, warning = sanitise_energy("0.6")
        assert val == 0.6
        assert warning is None

    def test_unparseable_string_defaults(self):
        val, warning = sanitise_energy("high")
        assert val == 0.5
        assert warning is not None

    def test_none_defaults(self):
        val, warning = sanitise_energy(None)
        assert val == 0.5
        assert warning is not None

    def test_integer_accepted(self):
        val, warning = sanitise_energy(1)
        assert val == 1.0
        assert warning is None


# ---------------------------------------------------------------------------
# 6. Valid genre / mood sets
# ---------------------------------------------------------------------------

class TestValidSets:
    def test_valid_genres_non_empty(self):
        assert len(VALID_GENRES) > 0

    def test_valid_moods_non_empty(self):
        assert len(VALID_MOODS) > 0

    def test_catalog_genres_in_valid_set(self):
        from src.recommender import load_songs
        songs = load_songs("data/songs.csv")
        catalog_genres = {s["genre"] for s in songs}
        missing = catalog_genres - VALID_GENRES
        assert not missing, f"Catalog genres missing from VALID_GENRES: {missing}"

    def test_catalog_moods_in_valid_set(self):
        from src.recommender import load_songs
        songs = load_songs("data/songs.csv")
        catalog_moods = {s["mood"] for s in songs}
        missing = catalog_moods - VALID_MOODS
        assert not missing, f"Catalog moods missing from VALID_MOODS: {missing}"
