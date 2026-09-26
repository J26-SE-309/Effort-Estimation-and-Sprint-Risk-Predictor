"""Text encoders E1-E4 behind one fit() / transform() / save() / load() interface (ML guide 3.2, 4.4 and 11.4).

E1 TF-IDF and E2 FastText learn from our own stories, so they are fitted on training text only (fitting on
test text is a classic leak) and saved with the models that use them: plain JSON and NumPy arrays, no pickles.
E3 SBERT is used as downloaded: its weights are not ours and are not data, so they live in the standard Hugging
Face cache of the machine, pinned to one exact revision. Its embeddings of the TAWOS stories are data, so they
are cached in the Datasets folder, keyed by a hash of the text: a story that has not changed is never encoded
twice (as the proposal plans). E4 (fine-tuned DistilBERT) is encoder and learner in one; see distilbert.py.
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from erp import config

EMBEDDINGS_DIR = config.WORK_DIR / "embeddings"
TOKEN = re.compile(r"[a-z0-9_]+")


def story_text(stories: pd.DataFrame) -> pd.Series:
    """What the encoders read: the title first (long descriptions are truncated), then the description."""
    return (stories["title"].fillna("") + ". " + stories["description_text"].fillna("")).str.strip()


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class TfidfEncoder:
    """E1: TF-IDF of words and word pairs, reduced to 300 values per story with a truncated SVD.

    The SVD keeps the input dense and small for learners that dislike 10,000 sparse columns (ML guide 4.4).
    Its components are kept in float32, in memory as on disk, so a reloaded encoder gives identical numbers.
    """

    name = "tfidf"
    dimensions = 300
    params = {"ngram_range": [1, 2], "sublinear_tf": True, "max_features": 10000, "min_df": 2, "svd": 300,
              "seed": 42}
    text = "title + '. ' + description_text (code removed); lower-cased words and word pairs"

    def __init__(self):
        self.vocabulary: list[str] = []
        self.idf: np.ndarray | None = None
        self.components: np.ndarray | None = None
        self._vectorizer = None

    def _make_vectorizer(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        p = self.params
        return TfidfVectorizer(ngram_range=tuple(p["ngram_range"]), sublinear_tf=p["sublinear_tf"],
                               max_features=p["max_features"], min_df=p["min_df"], dtype=np.float32)

    def fit(self, texts: pd.Series) -> "TfidfEncoder":
        from sklearn.decomposition import TruncatedSVD

        vectorizer = self._make_vectorizer()
        matrix = vectorizer.fit_transform(list(texts))
        svd = TruncatedSVD(self.params["svd"], random_state=self.params["seed"]).fit(matrix)
        self.vocabulary = vectorizer.get_feature_names_out().tolist()
        self.idf = vectorizer.idf_.astype(np.float32)
        self.components = svd.components_.astype(np.float32)
        self._vectorizer = None
        return self

    def _fitted_vectorizer(self):
        if self._vectorizer is None:
            vectorizer = self._make_vectorizer()
            vectorizer.vocabulary_ = {term: i for i, term in enumerate(self.vocabulary)}
            vectorizer.idf_ = self.idf
            self._vectorizer = vectorizer
        return self._vectorizer

    def transform(self, texts: pd.Series) -> np.ndarray:
        matrix = self._fitted_vectorizer().transform(list(texts))
        return np.asarray(matrix @ self.components.T, dtype=np.float32)

    def columns(self) -> list[str]:
        return [f"{self.name}_{i}" for i in range(self.dimensions)]

    def save(self, directory: Path) -> dict:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tfidf-vocabulary.json").write_text(json.dumps(self.vocabulary), encoding="utf-8")
        np.save(directory / "tfidf-idf.npy", self.idf)
        np.save(directory / "tfidf-svd.npy", self.components)
        return {"name": self.name, "params": self.params, "text": self.text,
                "files": ["tfidf-vocabulary.json", "tfidf-idf.npy", "tfidf-svd.npy"]}

    @classmethod
    def load(cls, directory: Path) -> "TfidfEncoder":
        encoder = cls()
        encoder.vocabulary = json.loads((directory / "tfidf-vocabulary.json").read_text(encoding="utf-8"))
        encoder.idf = np.load(directory / "tfidf-idf.npy", allow_pickle=False)
        encoder.components = np.load(directory / "tfidf-svd.npy", allow_pickle=False)
        return encoder


class FastTextEncoder:
    """E2: FastText word vectors trained on our training stories; a story is the mean of its word vectors.

    FastText builds words from character n-grams, so misspellings and words never seen in training still get a
    vector. The vectors are kept at half precision, in memory as on disk (about 20 MB instead of 45 MB); the
    story vector is computed exactly as gensim's get_vector does, from these arrays.
    """

    name = "fasttext"
    dimensions = 100
    params = {"vector_size": 100, "window": 5, "min_count": 2, "sg": 1, "epochs": 10, "bucket": 100_000,
              "min_n": 3, "max_n": 6, "seed": 42, "workers": 1}
    text = "title + '. ' + description_text (code removed); lower-cased word tokens"

    def __init__(self):
        self.words: list[str] = []
        self.vectors: np.ndarray | None = None  # final vector of every vocabulary word
        self.ngrams: np.ndarray | None = None  # vectors of the hashed character n-grams, for unseen words
        self._index: dict[str, int] = {}

    def fit(self, texts: pd.Series) -> "FastTextEncoder":
        from gensim.models import FastText

        p = self.params
        corpus = [tokens(t) for t in texts]
        model = FastText(vector_size=p["vector_size"], window=p["window"], min_count=p["min_count"], sg=p["sg"],
                         bucket=p["bucket"], min_n=p["min_n"], max_n=p["max_n"], seed=p["seed"],
                         workers=p["workers"])
        model.build_vocab(corpus)
        model.train(corpus, total_examples=len(corpus), epochs=p["epochs"])
        self.words = list(model.wv.index_to_key)
        self.vectors = model.wv.vectors.astype(np.float16)
        self.ngrams = model.wv.vectors_ngrams.astype(np.float16)
        self._index = {}
        return self

    def _word_vector(self, word: str, cache: dict) -> np.ndarray | None:
        from gensim.models.fasttext import ft_ngram_hashes

        if word not in cache:
            if not self._index:
                self._index = {w: i for i, w in enumerate(self.words)}
            if word in self._index:
                cache[word] = self.vectors[self._index[word]].astype(np.float32)
            else:
                hashes = ft_ngram_hashes(word, self.params["min_n"], self.params["max_n"], self.params["bucket"])
                cache[word] = self.ngrams[hashes].astype(np.float32).mean(axis=0) if hashes else None
        return cache[word]

    def transform(self, texts: pd.Series) -> np.ndarray:
        out = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        cache: dict = {}
        for row, text in enumerate(texts):
            vectors = [v for v in (self._word_vector(w, cache) for w in tokens(text)) if v is not None]
            if vectors:
                out[row] = np.mean(vectors, axis=0)
        return out

    def columns(self) -> list[str]:
        return [f"{self.name}_{i}" for i in range(self.dimensions)]

    def save(self, directory: Path) -> dict:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "fasttext-words.json").write_text(json.dumps(self.words), encoding="utf-8")
        np.save(directory / "fasttext-vectors.npy", self.vectors)
        np.save(directory / "fasttext-ngrams.npy", self.ngrams)
        return {"name": self.name, "params": self.params, "text": self.text,
                "files": ["fasttext-words.json", "fasttext-vectors.npy", "fasttext-ngrams.npy"]}

    @classmethod
    def load(cls, directory: Path) -> "FastTextEncoder":
        encoder = cls()
        encoder.words = json.loads((directory / "fasttext-words.json").read_text(encoding="utf-8"))
        encoder.vectors = np.load(directory / "fasttext-vectors.npy", allow_pickle=False)
        encoder.ngrams = np.load(directory / "fasttext-ngrams.npy", allow_pickle=False)
        return encoder


class SbertEncoder:
    """E3: SBERT all-MiniLM-L6-v2 (384 values per text), used as downloaded, on the CPU."""

    name = "sbert"
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    revision = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"  # the Hugging Face commit of the weights we use
    dimensions = 384
    text = "title + '. ' + description_text (code removed), truncated to 256 word pieces"

    def __init__(self, store: Path | None = None, model=None, batch_size: int = 64):
        self.store = store or EMBEDDINGS_DIR / "sbert-all-MiniLM-L6-v2.npz"
        self._model = model
        self.batch_size = batch_size

    def fit(self, texts: pd.Series) -> "SbertEncoder":
        return self  # pre-trained: nothing to learn from our data

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id, revision=self.revision, device="cpu")
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode without the cache (what the service does for a story it has never seen)."""
        vectors = self._load_model().encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                            show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    def _cached(self) -> dict[str, np.ndarray]:
        if not self.store.exists():
            return {}
        data = np.load(self.store)
        return dict(zip(data["hashes"].tolist(), data["vectors"], strict=True))

    def transform(self, texts: pd.Series) -> np.ndarray:
        hashes = [text_hash(t) for t in texts]
        cache = self._cached()
        missing = {h: t for h, t in zip(hashes, texts, strict=True) if h not in cache}
        if missing:
            cache.update(zip(missing.keys(), self.encode(list(missing.values())), strict=True))
            self.store.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.store, hashes=np.array(list(cache.keys())), vectors=np.stack(list(cache.values())))
        return np.stack([cache[h] for h in hashes]).astype(np.float32)

    def columns(self) -> list[str]:
        return [f"{self.name}_{i}" for i in range(self.dimensions)]

    def save(self, directory: Path) -> dict:
        return {"name": self.name, "model_id": self.model_id, "revision": self.revision,
                "dimensions": self.dimensions, "text": self.text, "files": []}

    @classmethod
    def load(cls, directory: Path | None = None) -> "SbertEncoder":
        return cls()


ENCODERS = {encoder.name: encoder for encoder in (TfidfEncoder, FastTextEncoder, SbertEncoder)}
