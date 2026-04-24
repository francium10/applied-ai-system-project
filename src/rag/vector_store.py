"""
src/rag/vector_store.py
========================
In-memory vector store for song embeddings.

Stores pre-computed song embeddings and supports nearest-neighbour
search by cosine similarity. Designed to be swapped out for a
production store (ChromaDB, Pinecone) without changing the retriever.

Interface contract:
  build(songs)           -> None   index all songs
  search(query_vec, k)   -> list   return k most similar song dicts
  size()                 -> int    number of indexed songs
"""

from typing import Dict, List

from src.rag.embedder import cosine_similarity, embed_song


class VectorStore:
    """
    In-memory nearest-neighbour store using cosine similarity.

    Usage
    -----
    >>> store = VectorStore()
    >>> store.build(songs)
    >>> results = store.search(query_vector, k=5)
    """

    def __init__(self) -> None:
        # List of (song_dict, embedding_vector) pairs
        self._index: List[tuple] = []

    def build(self, songs: List[Dict]) -> None:
        """
        Embed all songs and store them in the index.

        Calling build() again clears and rebuilds the full index.
        This is intentional: when the catalog changes, call build() again.
        """
        self._index = []
        for song in songs:
            vector = embed_song(song)
            self._index.append((song, vector))

    def search(self, query_vector: List[float], k: int = 5) -> List[Dict]:
        """
        Return the k songs most similar to query_vector.

        Scores every entry by cosine similarity then returns the top-k
        song dicts sorted by similarity descending.

        Parameters
        ----------
        query_vector : Embedding from embed_query().
        k            : Number of results to return.

        Returns
        -------
        List of song dicts (not tuples — callers get plain dicts).
        """
        if not self._index:
            return []

        scored = [
            (song, cosine_similarity(query_vector, vector))
            for song, vector in self._index
        ]

        # Sort by similarity descending; ties broken by song id
        scored.sort(key=lambda pair: (-pair[1], pair[0].get("id", 0)))
        return [song for song, _ in scored[:k]]

    def search_with_scores(
        self, query_vector: List[float], k: int = 5
    ) -> List[tuple]:
        """
        Return (song, similarity_score) tuples for the top-k results.

        Used by the retriever when the similarity score is needed
        downstream (e.g. for confidence weighting).
        """
        if not self._index:
            return []

        scored = [
            (song, cosine_similarity(query_vector, vector))
            for song, vector in self._index
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0].get("id", 0)))
        return scored[:k]

    def size(self) -> int:
        """Return the number of songs currently indexed."""
        return len(self._index)
