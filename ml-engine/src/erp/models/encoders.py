"""Text encoders E1-E4 behind one fit() / transform() interface (ML guide 3.2 and 11.4).

Only E3 exists so far. Its embeddings of the TAWOS stories are data, so they are cached in the Datasets
folder, keyed by a hash of the text: a story that has not changed is never encoded twice (as the proposal
plans). The downloaded SBERT weights are not ours and are not data: they live in the standard Hugging Face
cache of the machine, pinned to one exact revision so every run uses the same model.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from erp import config

EMBEDDINGS_DIR = config.WORK_DIR / "embeddings"


def story_text(stories: pd.DataFrame) -> pd.Series:
    """What the encoders read: the title first (long descriptions are truncated), then the description."""
    return (stories["title"].fillna("") + ". " + stories["description_text"].fillna("")).str.strip()


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


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
            vectors = self._load_model().encode(list(missing.values()), batch_size=self.batch_size,
                                                normalize_embeddings=True, show_progress_bar=False)
            cache.update(zip(missing.keys(), np.asarray(vectors, dtype=np.float32), strict=True))
            self.store.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.store, hashes=np.array(list(cache.keys())), vectors=np.stack(list(cache.values())))
        return np.stack([cache[h] for h in hashes]).astype(np.float32)

    def columns(self) -> list[str]:
        return [f"{self.name}_{i}" for i in range(self.dimensions)]
