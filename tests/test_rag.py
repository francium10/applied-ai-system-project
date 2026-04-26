"""
tests/test_rag.py
==================
Full test suite for the RAG pipeline: embedder, vector store, and retriever.

Coverage map:
  embedder.py    — embed_song, embed_query, cosine_similarity, _vectorise
  vector_store.py— build, search, search_with_scores, size, empty store
  retriever.py   — retrieve, retrieve_with_scores, top_similarity

Run with:  pytest tests/test_rag.py -v
"""

import math
import pytest
from src.rag.embedder import (
    embed_song, embed_query, cosine_similarity,
    VOCAB_SIZE, _TOKEN_INDEX,
)
from src.rag.vector_store import VectorStore
from src.rag.retriever import Retriever
from src.recommender import load_songs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def songs():
    return load_songs("data/songs.csv")


@pytest.fixture(scope="session")
def built_store(songs):
    store = VectorStore()
    store.build(songs)
    return store


@pytest.fixture(scope="session")
def retriever(built_store):
    return Retriever(built_store)


# ---------------------------------------------------------------------------
# 1. Embedder — embed_song
# ---------------------------------------------------------------------------

class TestEmbedSong:
    def test_returns_list_of_correct_length(self, songs):
        vec = embed_song(songs[0])
        assert isinstance(vec, list)
        assert len(vec) == VOCAB_SIZE

    def test_all_values_are_floats(self, songs):
        vec = embed_song(songs[0])
        assert all(isinstance(v, float) for v in vec)

    def test_vector_is_l2_normalised(self, songs):
        """L2 norm should be ~1.0 for any non-zero song embedding."""
        vec = embed_song(songs[0])
        norm = math.sqrt(sum(v * v for v in vec))
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_different_songs_produce_different_vectors(self, songs):
        pop = next(s for s in songs if s["genre"] == "pop")
        lofi = next(s for s in songs if s["genre"] == "lofi")
        assert embed_song(pop) != embed_song(lofi)

    def test_same_song_produces_same_vector(self, songs):
        assert embed_song(songs[0]) == embed_song(songs[0])

    def test_genre_token_present_in_vector(self, songs):
        pop_song = next(s for s in songs if s["genre"] == "pop")
        vec = embed_song(pop_song)
        idx = _TOKEN_INDEX.get("pop")
        assert idx is not None and vec[idx] > 0

    def test_empty_song_returns_sparse_vector(self):
        """Empty genre/mood/tags song still gets energy descriptor tokens."""
        empty_song = {"genre": "", "mood": "", "mood_tags": "", "energy": 0.5}
        vec = embed_song(empty_song)
        # energy=0.5 → "medium energy" → at least one token active
        assert any(
            v > 0 for v in vec), "Expected energy descriptor to activate at least one token"


# ---------------------------------------------------------------------------
# 2. Embedder — embed_query
# ---------------------------------------------------------------------------

class TestEmbedQuery:
    def test_returns_correct_length(self):
        assert len(embed_query("lofi chill focused")) == VOCAB_SIZE

    def test_all_values_floats(self):
        vec = embed_query("pop happy euphoric")
        assert all(isinstance(v, float) for v in vec)

    def test_known_token_activates_index(self):
        vec = embed_query("pop")
        idx = _TOKEN_INDEX["pop"]
        assert vec[idx] > 0

    def test_empty_query_returns_zero_vector(self):
        vec = embed_query("")
        assert all(v == 0.0 for v in vec)

    def test_unknown_words_ignored(self):
        # Words not in vocab contribute nothing
        vec_unknown = embed_query("xyzzy frobozz quux")
        assert all(v == 0.0 for v in vec_unknown)

    def test_query_normalised(self):
        vec = embed_query("pop happy")
        norm = math.sqrt(sum(v * v for v in vec))
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_case_insensitive(self):
        assert embed_query("POP HAPPY") == embed_query("pop happy")


# ---------------------------------------------------------------------------
# 3. Embedder — cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        v = embed_query("pop happy euphoric")
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_zero_vector_returns_zero(self):
        zero = [0.0] * VOCAB_SIZE
        other = embed_query("pop")
        assert cosine_similarity(zero, other) == 0.0
        assert cosine_similarity(other, zero) == 0.0

    def test_both_zero_returns_zero(self):
        zero = [0.0] * VOCAB_SIZE
        assert cosine_similarity(zero, zero) == 0.0

    def test_similar_queries_higher_than_dissimilar(self):
        base = embed_query("lofi chill")
        similar = embed_query("lofi focused chill")
        differ = embed_query("metal angry intense")
        sim_score = cosine_similarity(base, similar)
        diff_score = cosine_similarity(base, differ)
        assert sim_score > diff_score

    def test_symmetric(self):
        a = embed_query("pop happy")
        b = embed_query("lofi chill")
        assert cosine_similarity(a, b) == pytest.approx(
            cosine_similarity(b, a), abs=1e-9)


# ---------------------------------------------------------------------------
# 4. VectorStore
# ---------------------------------------------------------------------------

class TestVectorStore:
    def test_size_after_build(self, built_store, songs):
        assert built_store.size() == len(songs)

    def test_empty_store_search_returns_empty(self):
        empty = VectorStore()
        assert empty.search(embed_query("pop"), k=5) == []

    def test_empty_store_search_with_scores_returns_empty(self):
        empty = VectorStore()
        assert empty.search_with_scores(embed_query("pop"), k=5) == []

    def test_search_returns_list_of_dicts(self, built_store):
        results = built_store.search(embed_query("pop happy"), k=3)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, dict)

    def test_search_returns_correct_k(self, built_store):
        results = built_store.search(embed_query("pop"), k=4)
        assert len(results) == 4

    def test_search_k_larger_than_catalog_returns_all(self, built_store, songs):
        results = built_store.search(embed_query("pop"), k=999)
        assert len(results) == len(songs)

    def test_search_with_scores_returns_tuples(self, built_store):
        results = built_store.search_with_scores(
            embed_query("lofi chill"), k=3)
        for song, score in results:
            assert isinstance(song, dict)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_search_with_scores_sorted_descending(self, built_store):
        results = built_store.search_with_scores(
            embed_query("lofi chill"), k=5)
        scores = [sc for _, sc in results]
        assert scores == sorted(scores, reverse=True)

    def test_rebuild_clears_old_index(self, songs):
        store = VectorStore()
        store.build(songs)
        size_before = store.size()
        store.build(songs[:5])
        assert store.size() == 5
        assert store.size() != size_before

    def test_size_zero_before_build(self):
        store = VectorStore()
        assert store.size() == 0


# ---------------------------------------------------------------------------
# 5. Retriever
# ---------------------------------------------------------------------------

class TestRetriever:
    def test_retrieve_returns_list(self, retriever):
        results = retriever.retrieve("lofi chill focused", k=5)
        assert isinstance(results, list)

    def test_retrieve_returns_k_songs(self, retriever):
        results = retriever.retrieve("pop happy", k=4)
        assert len(results) == 4

    def test_lofi_query_surfaces_lofi_songs(self, retriever):
        results = retriever.retrieve("lofi chill focused", k=5)
        genres = [s["genre"] for s in results]
        assert "lofi" in genres, f"Expected lofi in top results, got {genres}"

    def test_pop_query_surfaces_pop_songs(self, retriever):
        results = retriever.retrieve("pop happy euphoric", k=5)
        genres = [s["genre"] for s in results]
        assert "pop" in genres or "indie pop" in genres

    def test_metal_query_surfaces_metal_song(self, retriever):
        # Metal is 1 of 20 genres; use wider k to guarantee it appears
        results = retriever.retrieve("metal angry intense aggressive", k=8)
        genres = [s["genre"] for s in results]
        assert "metal" in genres, f"Expected metal in top-8 results, got {genres}"

    def test_retrieve_with_scores_returns_tuples(self, retriever):
        results = retriever.retrieve_with_scores("pop happy", k=3)
        for song, score in results:
            assert isinstance(song, dict)
            assert 0.0 <= score <= 1.0

    def test_retrieve_with_scores_sorted_descending(self, retriever):
        results = retriever.retrieve_with_scores("lofi chill", k=5)
        scores = [sc for _, sc in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_similarity_returns_float(self, retriever):
        sim = retriever.top_similarity("pop happy")
        assert isinstance(sim, float)
        assert 0.0 <= sim <= 1.0

    def test_top_similarity_clear_query_higher_than_ambiguous(self, retriever):
        clear = retriever.top_similarity("lofi chill focused mellow")
        ambiguous = retriever.top_similarity("good music")
        assert clear >= ambiguous

    def test_top_similarity_empty_store_returns_zero(self):
        empty_store = VectorStore()
        r = Retriever(empty_store)
        assert r.top_similarity("pop happy") == 0.0

    def test_k_larger_than_catalog_returns_all(self, retriever, songs):
        results = retriever.retrieve("pop", k=999)
        assert len(results) == len(songs)
