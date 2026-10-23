"""The networks trained in PyTorch, run with ONNX Runtime: what the prediction service image uses instead of PyTorch.

`erp-export-onnx` (erp.models.onnx_export) converts SBERT's encoder and the fine-tuned DistilBERT networks to ONNX
when the image is built, and checks that ONNX Runtime reproduces PyTorch's outputs. Each export sits beside the
model it replaces, with the tokenizer it needs (the `tokenizers` library, which reads the same tokenizer.json the
Hugging Face tokenizers use, with the same truncation and padding). Training and the arena still use PyTorch, and
so does a machine without an export.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np

TOKENIZER = "tokenizer.json"
PACKAGES = ("onnxruntime", "tokenizers")


def ready(*files: Path) -> bool:
    """True when the exported files are present and ONNX Runtime and tokenizers are installed."""
    return (all(importlib.util.find_spec(package) is not None for package in PACKAGES)
            and all(Path(f).exists() for f in files))


def session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    # Each run frees its buffers, as PyTorch does. ONNX Runtime's arena otherwise keeps the largest it ever needed:
    # inputs change length with every batch of stories, and two workers outgrew the 4 GB limit (NFR12).
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    # One native thread per worker, as OMP_NUM_THREADS gives LightGBM and OpenBLAS (see backend/Dockerfile):
    # ONNX Runtime does not read that variable and would otherwise start a thread per core in every worker.
    threads = os.environ.get("OMP_NUM_THREADS")
    if threads:
        options.intra_op_num_threads = options.inter_op_num_threads = int(threads)
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def tokenizer(path: Path):
    """The exported tokenizer, with the truncation and padding it was saved with."""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def tokenize(tok, texts: list[str]) -> dict[str, np.ndarray]:
    encoded = tok.encode_batch(texts)
    return {"input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encoded], dtype=np.int64)}


class SbertModel:
    """SBERT's encoder with its mean pooling and normalisation in one graph: encode() returns what
    SentenceTransformer.encode(..., normalize_embeddings=True) does."""

    MODEL = "model.onnx"

    def __init__(self, directory: Path, dimensions: int):
        self.session = session(directory / self.MODEL)
        self.tokenizer = tokenizer(directory / TOKENIZER)
        self.dimensions = dimensions

    def encode(self, texts: list[str], batch_size: int = 64, **_) -> np.ndarray:
        out = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        # longest texts first, as sentence-transformers does, so a batch pads to similar lengths
        order = np.argsort([-len(t) for t in texts], kind="stable")
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            out[rows] = self.session.run(None, tokenize(self.tokenizer, [texts[i] for i in rows]))[0]
        return out


class DistilBertNet:
    """One fine-tuned DistilBERT network (encoder and head): the raw output for each story, as the PyTorch network
    gives it (standardised effort or the risk logit)."""

    def __init__(self, path: Path, tokenizer_path: Path):
        self.session = session(path)
        self.tokenizer = tokenizer(tokenizer_path)

    def forward(self, texts: list[str], features: np.ndarray, size: int = 64) -> np.ndarray:
        out = np.zeros(len(texts), dtype=np.float32)
        for start in range(0, len(texts), size):
            rows = np.arange(start, min(start + size, len(texts)))
            inputs = tokenize(self.tokenizer, [texts[i] for i in rows])
            del inputs["token_type_ids"]  # DistilBERT has no segment embeddings
            inputs["features"] = np.ascontiguousarray(features[rows], dtype=np.float32)
            out[rows] = self.session.run(None, inputs)[0]
        return out
