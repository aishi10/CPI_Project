"""
Similarity scorers (0-1) used for (a) routing a statement to a memory (Part 1)
and (b) the relevance gate in front of the classifier (Part 2).

    get_similarity("words")  -> stopword-filtered Jaccard overlap (no dependencies, offline)
    get_similarity("embed")  -> cosine similarity of sentence embeddings (all-MiniLM-L6-v2)

DEFAULT_MIN_RELEVANCE are starting points; tune them on your synthetic dataset.
"""
import re

_STOP = {"the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "at", "on", "and", "or", "for", "with"}

DEFAULT_MIN_RELEVANCE = {"words": 0.25, "embed": 0.35}


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP}


def jaccard_similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class EmbeddingSimilarity:
    """Cosine similarity of normalised sentence embeddings. Model loads on first use; vectors are cached."""
    MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        self._model = None
        self._cache = {}

    def _vec(self, text: str):
        if text not in self._cache:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as e:
                    raise ImportError("Embedding similarity needs sentence-transformers: "
                                      "pip install sentence-transformers  (or use --similarity words)") from e
                self._model = SentenceTransformer(self.MODEL)
            self._cache[text] = self._model.encode(text, normalize_embeddings=True)
        return self._cache[text]

    def __call__(self, a: str, b: str) -> float:
        dot = sum(float(x) * float(y) for x, y in zip(self._vec(a), self._vec(b)))
        return max(0.0, min(1.0, dot))


def get_similarity(name: str):
    if name == "words":
        return jaccard_similarity
    if name == "embed":
        return EmbeddingSimilarity()
    raise ValueError(f"unknown similarity {name!r}; use 'words' or 'embed'")
