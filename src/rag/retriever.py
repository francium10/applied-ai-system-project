"""
src/rag/retriever.py
=====================
RAG retriever — bridges natural language queries and the scoring engine.

The retriever's job is to take whatever the user typed and return the
subset of songs most likely to be relevant, so the scoring engine
doesn't need to score all 20 songs for every query.

For a 20-song catalog this is technically unnecessary, but it models
what a real RAG pipeline does: retrieve a relevant candidate set first,
then apply more expensive ranking logic only to those candidates.

Public API
----------
  Retriever(store)          build from an existing VectorStore
  retrieve(query, k)   ->   list[dict]   top-k song dicts
  retrieve_with_scores(q,k) ->   list[(dict, float)]
"""

from typing import Dict, List, Tuple

from src.rag.embedder import embed_query
from src.rag.vector_store import VectorStore


class Retriever:
    """
    Semantic song retriever backed by a VectorStore.

    Usage
    -----
    >>> store = VectorStore()
    >>> store.build(songs)
    >>> retriever = Retriever(store)
    >>> candidates = retriever.retrieve("something euphoric for a run", k=8)
    """

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def retrieve(self, query: str, k: int = 10) -> List[Dict]:
        """
        Embed the query and return the k most similar songs.

        We default k=10 (half the catalog) so the scoring engine has
        enough candidates for diversity while still filtering out the
        least relevant songs.

        Parameters
        ----------
        query : Natural language string or structured text representation.
        k     : Number of candidate songs to return.

        Returns
        -------
        List of song dicts sorted by semantic similarity descending.
        """
        query_vector = embed_query(query)
        return self._store.search(query_vector, k=k)

    def retrieve_with_scores(
        self, query: str, k: int = 10
    ) -> List[Tuple[Dict, float]]:
        """
        Return (song, similarity_score) pairs for the top-k results.

        The similarity score (0–1) is used by the planner to weight
        confidence: if the top candidate has low similarity, the planner
        knows the query was ambiguous and lowers confidence accordingly.
        """
        query_vector = embed_query(query)
        return self._store.search_with_scores(query_vector, k=k)

    def top_similarity(self, query: str) -> float:
        """
        Return the highest cosine similarity score for any song.

        Used by the critic to assess how well the query matched the catalog.
        A score below 0.3 signals an ambiguous or off-catalog query.
        """
        results = self.retrieve_with_scores(query, k=1)
        if not results:
            return 0.0
        _, score = results[0]
        return round(score, 4)
