"""
tests/test_agent.py
====================
Full test suite for the agentic layer: critic, planner, and logger.

Coverage map:
  critic.py    — CritiqueResult dataclass, offline critique signals,
                 label thresholds, should_rerank logic
  planner.py   — run_pipeline happy path, guardrail abort, re-rank trigger,
                 PipelineResult fields, natural language input
  logger.py    — log_decision writes, log_error writes, read_log reads,
                 clear_log deletes, malformed lines skipped

Run with:  pytest tests/test_agent.py -v
"""

import json
import os
import tempfile
import pytest
from pathlib import Path

from src.agent.critic import critique, CritiqueResult, RERANK_THRESHOLD
from src.agent.planner import run_pipeline, PipelineResult
from src.reliability.logger import log_decision, log_error, read_log, clear_log
from src.recommender import load_songs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def songs():
    return load_songs("data/songs.csv")

@pytest.fixture
def pop_profile():
    return {"genre": "pop", "mood": "happy", "energy": 0.8,
            "likes_acoustic": False, "preferred_mood_tags": [], "preferred_decade": ""}

@pytest.fixture
def lofi_profile():
    return {"genre": "lofi", "mood": "chill", "energy": 0.4,
            "likes_acoustic": True, "preferred_mood_tags": [], "preferred_decade": ""}

@pytest.fixture
def good_results(songs):
    """Top-3 results all matching pop/happy — high confidence case."""
    pop = [s for s in songs if s["genre"] == "pop" and s["mood"] == "happy"]
    others = [s for s in songs if s not in pop][:1]
    top = (pop + others)[:3]
    return [(s, 9.5 - i * 0.3, "genre match; mood match") for i, s in enumerate(top)]

@pytest.fixture
def bad_results(songs):
    """Top-3 results all genre/mood mismatches — low confidence case."""
    # Pick songs that clearly don't match pop/happy
    mismatches = [s for s in songs if s["genre"] != "pop" and s["mood"] != "happy"][:3]
    return [(s, 3.0 - i * 0.2, "General catalog suggestion") for i, s in enumerate(mismatches)]

@pytest.fixture(autouse=True)
def clean_logs():
    """Clear the log file before and after every test."""
    clear_log()
    yield
    clear_log()


# ---------------------------------------------------------------------------
# 1. CritiqueResult dataclass
# ---------------------------------------------------------------------------

class TestCritiqueResult:
    def test_label_high(self):
        r = CritiqueResult(confidence=0.90, note="good", should_rerank=False)
        assert r.label() == "HIGH"

    def test_label_medium(self):
        r = CritiqueResult(confidence=0.65, note="ok", should_rerank=False)
        assert r.label() == "MEDIUM"

    def test_label_low(self):
        r = CritiqueResult(confidence=0.40, note="poor", should_rerank=True)
        assert r.label() == "LOW"

    def test_label_boundary_medium_lower(self):
        r = CritiqueResult(confidence=0.55, note="borderline", should_rerank=False)
        assert r.label() == "MEDIUM"

    def test_label_boundary_high_lower(self):
        r = CritiqueResult(confidence=0.80, note="high boundary", should_rerank=False)
        assert r.label() == "HIGH"


# ---------------------------------------------------------------------------
# 2. Offline critic — confidence scoring
# ---------------------------------------------------------------------------

class TestOfflineCritic:
    def test_returns_critique_result(self, good_results, pop_profile):
        result = critique("happy pop", pop_profile, good_results, top_rag_similarity=0.5)
        assert isinstance(result, CritiqueResult)

    def test_confidence_in_range(self, good_results, pop_profile):
        result = critique("happy pop", pop_profile, good_results, top_rag_similarity=0.5)
        assert 0.0 <= result.confidence <= 1.0

    def test_note_is_non_empty_string(self, good_results, pop_profile):
        result = critique("happy pop", pop_profile, good_results, top_rag_similarity=0.5)
        assert isinstance(result.note, str) and result.note.strip() != ""

    def test_should_rerank_is_bool(self, good_results, pop_profile):
        result = critique("happy pop", pop_profile, good_results, top_rag_similarity=0.5)
        assert isinstance(result.should_rerank, bool)

    def test_good_results_high_confidence(self, good_results, pop_profile):
        result = critique("happy pop", pop_profile, good_results, top_rag_similarity=0.6)
        assert result.confidence >= 0.50

    def test_bad_results_lower_confidence(self, good_results, bad_results, pop_profile):
        good_c = critique("happy pop", pop_profile, good_results, top_rag_similarity=0.6).confidence
        bad_c  = critique("happy pop", pop_profile, bad_results,  top_rag_similarity=0.1).confidence
        assert good_c > bad_c

    def test_high_rag_similarity_boosts_confidence(self, good_results, pop_profile):
        low_sim  = critique("pop", pop_profile, good_results, top_rag_similarity=0.05).confidence
        high_sim = critique("pop happy", pop_profile, good_results, top_rag_similarity=0.60).confidence
        assert high_sim > low_sim

    def test_empty_results_returns_zero_confidence(self, pop_profile):
        result = critique("pop", pop_profile, [], top_rag_similarity=0.5)
        assert result.confidence == 0.0

    def test_should_rerank_true_when_low_confidence(self, bad_results, pop_profile):
        result = critique("pop", pop_profile, bad_results, top_rag_similarity=0.05)
        if result.confidence < RERANK_THRESHOLD:
            assert result.should_rerank is True

    def test_should_rerank_false_when_high_confidence(self, good_results, pop_profile):
        result = critique("pop happy", pop_profile, good_results, top_rag_similarity=0.7)
        if result.confidence >= RERANK_THRESHOLD:
            assert result.should_rerank is False

    def test_offline_note_contains_offline_marker(self, good_results, pop_profile):
        # Without API key, note should indicate offline mode
        result = critique("pop", pop_profile, good_results, top_rag_similarity=0.5)
        assert "offline" in result.note.lower()

    def test_attempt_two_skips_online_critic(self, good_results, pop_profile):
        """Second attempt always uses offline critic regardless of env."""
        result = critique("pop", pop_profile, good_results,
                          top_rag_similarity=0.5, attempt=2)
        assert isinstance(result, CritiqueResult)


# ---------------------------------------------------------------------------
# 3. run_pipeline — happy paths
# ---------------------------------------------------------------------------

class TestRunPipeline:
    def test_returns_pipeline_result(self, songs, pop_profile):
        result = run_pipeline(pop_profile, k=3, songs=songs)
        assert isinstance(result, PipelineResult)

    def test_recommendations_count(self, songs, pop_profile):
        result = run_pipeline(pop_profile, k=3, songs=songs)
        assert len(result.recommendations) == 3

    def test_each_recommendation_three_tuple(self, songs, pop_profile):
        result = run_pipeline(pop_profile, k=3, songs=songs)
        for song, score, explanation in result.recommendations:
            assert isinstance(song, dict)
            assert isinstance(score, float)
            assert isinstance(explanation, str)

    def test_critique_is_critique_result(self, songs, pop_profile):
        result = run_pipeline(pop_profile, k=3, songs=songs)
        assert isinstance(result.critique, CritiqueResult)

    def test_confidence_in_range(self, songs, pop_profile):
        result = run_pipeline(pop_profile, k=5, songs=songs)
        assert 0.0 <= result.critique.confidence <= 1.0

    def test_rerank_triggered_is_bool(self, songs, pop_profile):
        result = run_pipeline(pop_profile, k=3, songs=songs)
        assert isinstance(result.rerank_triggered, bool)

    def test_mode_used_is_scoring_mode(self, songs, pop_profile):
        from src.recommender import ScoringMode
        result = run_pipeline(pop_profile, k=3, songs=songs)
        assert isinstance(result.mode_used, ScoringMode)

    def test_guardrail_errors_is_list(self, songs, pop_profile):
        result = run_pipeline(pop_profile, k=3, songs=songs)
        assert isinstance(result.guardrail_errors, list)

    def test_natural_language_query_works(self, songs):
        result = run_pipeline("something chill for late night studying", k=3, songs=songs)
        assert isinstance(result, PipelineResult)
        assert len(result.recommendations) == 3

    def test_lofi_profile_top_result_is_lofi(self, songs, lofi_profile):
        result = run_pipeline(lofi_profile, k=5, songs=songs)
        top_song, _, _ = result.recommendations[0]
        assert top_song["genre"] == "lofi"

    def test_k_respected(self, songs, pop_profile):
        for k in [1, 3, 5]:
            result = run_pipeline(pop_profile, k=k, songs=songs)
            assert len(result.recommendations) == k

    def test_str_output_non_empty(self, songs, pop_profile):
        result = run_pipeline(pop_profile, k=3, songs=songs)
        assert str(result).strip() != ""

    def test_log_written_after_pipeline(self, songs, pop_profile):
        clear_log()
        run_pipeline(pop_profile, k=3, songs=songs)
        records = read_log(n=1)
        assert len(records) == 1


# ---------------------------------------------------------------------------
# 4. run_pipeline — guardrail aborts
# ---------------------------------------------------------------------------

class TestPipelineGuardrails:
    def test_empty_genre_raises_value_error(self, songs):
        with pytest.raises(ValueError, match="GUARDRAIL"):
            run_pipeline({"genre": "", "mood": "happy", "energy": 0.5}, songs=songs)

    def test_none_mood_raises_value_error(self, songs):
        with pytest.raises(ValueError, match="GUARDRAIL"):
            run_pipeline({"genre": "pop", "mood": None, "energy": 0.5}, songs=songs)

    def test_both_empty_raises_value_error(self, songs):
        with pytest.raises(ValueError, match="GUARDRAIL"):
            run_pipeline({"genre": "", "mood": None, "energy": 0.5}, songs=songs)

    def test_guardrail_abort_writes_error_log(self, songs):
        clear_log()
        try:
            run_pipeline({"genre": "", "mood": None, "energy": 0.5}, songs=songs)
        except ValueError:
            pass
        from pathlib import Path
        log_path = Path("logs/decisions.json")
        assert log_path.exists()

    def test_soft_error_doesnt_abort(self, songs):
        """Unknown genre is a soft warning — pipeline should still run."""
        result = run_pipeline({"genre": "bossa nova", "mood": "relaxed",
                               "energy": 0.5, "likes_acoustic": True,
                               "preferred_mood_tags": [], "preferred_decade": ""},
                              k=3, songs=songs)
        assert isinstance(result, PipelineResult)
        assert len(result.recommendations) == 3


# ---------------------------------------------------------------------------
# 5. Logger
# ---------------------------------------------------------------------------

class TestLogger:
    def test_log_decision_creates_file(self):
        clear_log()
        log_decision(
            query="test query", parsed_profile={"genre": "pop"},
            rag_candidates=[1, 2, 3],
            top_results=[{"id": 1, "title": "Song A", "score": 9.0}],
            confidence=0.85, rerank_triggered=False,
            critic_note="Good match", mode="balanced",
        )
        assert Path("logs/decisions.json").exists()

    def test_log_decision_record_readable(self):
        clear_log()
        log_decision(
            query="pop happy run", parsed_profile={"genre": "pop", "mood": "happy"},
            rag_candidates=[1, 2], top_results=[{"id": 1, "title": "T", "score": 8.0}],
            confidence=0.88, rerank_triggered=False, critic_note="Strong match",
        )
        records = read_log(n=1)
        assert len(records) == 1
        rec = records[0]
        assert rec["query"] == "pop happy run"
        assert rec["confidence"] == pytest.approx(0.88, abs=0.01)

    def test_log_decision_fields_present(self):
        clear_log()
        log_decision(
            query="test", parsed_profile={}, rag_candidates=[],
            top_results=[], confidence=0.5, rerank_triggered=False,
            critic_note="note", guardrail_errors=["warn"], mode="mood_first",
        )
        rec = read_log(n=1)[0]
        for field in ["timestamp", "query", "mode", "parsed_profile",
                      "rag_candidates", "top_results", "confidence",
                      "rerank_triggered", "critic_note", "guardrail_errors"]:
            assert field in rec, f"Missing field: {field}"

    def test_log_error_creates_record(self):
        clear_log()
        log_error({"genre": ""}, ["genre: cannot be empty"])
        records = read_log(n=1)
        assert len(records) == 1
        assert records[0]["status"] == "rejected"

    def test_multiple_records_appended(self):
        clear_log()
        for i in range(3):
            log_decision(
                query=f"query {i}", parsed_profile={}, rag_candidates=[],
                top_results=[], confidence=0.5 + i * 0.1,
                rerank_triggered=False, critic_note="note",
            )
        records = read_log(n=10)
        assert len(records) == 3

    def test_read_log_empty_when_no_file(self):
        clear_log()
        records = read_log(n=5)
        assert records == []

    def test_read_log_n_limits_results(self):
        clear_log()
        for i in range(5):
            log_decision(
                query=f"q{i}", parsed_profile={}, rag_candidates=[],
                top_results=[], confidence=0.5, rerank_triggered=False, critic_note="",
            )
        records = read_log(n=3)
        assert len(records) == 3

    def test_clear_log_removes_file(self):
        log_decision(
            query="x", parsed_profile={}, rag_candidates=[], top_results=[],
            confidence=0.5, rerank_triggered=False, critic_note="",
        )
        clear_log()
        assert not Path("logs/decisions.json").exists()

    def test_confidence_rounded_in_log(self):
        clear_log()
        log_decision(
            query="test", parsed_profile={}, rag_candidates=[], top_results=[],
            confidence=0.123456789, rerank_triggered=False, critic_note="",
        )
        rec = read_log(n=1)[0]
        # Should be rounded to 4 decimal places
        assert len(str(rec["confidence"]).split(".")[-1]) <= 4

    def test_rerank_flag_stored_correctly(self):
        clear_log()
        log_decision(
            query="test", parsed_profile={}, rag_candidates=[], top_results=[],
            confidence=0.4, rerank_triggered=True, critic_note="low confidence",
        )
        rec = read_log(n=1)[0]
        assert rec["rerank_triggered"] is True

    def test_malformed_json_line_skipped(self):
        """read_log should skip broken lines without crashing."""
        Path("logs").mkdir(exist_ok=True)
        with open("logs/decisions.json", "w") as f:
            f.write('{"query": "good"}\n')
            f.write("NOT VALID JSON\n")
            f.write('{"query": "also good"}\n')
        records = read_log(n=10)
        assert len(records) == 2
        assert all("query" in r for r in records)
