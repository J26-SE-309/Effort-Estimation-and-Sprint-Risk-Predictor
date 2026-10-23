"""erp-export-onnx: the networks the service would run in PyTorch, converted to ONNX for an image without PyTorch.

  SBERT (E3)        <arena>/encoders/sbert/model.onnx + tokenizer.json: the transformer with SBERT's mean pooling
                    and normalisation, so it returns what SentenceTransformer.encode returns
  DistilBERT (E4)   <arena>/distilbert/effort.onnx, risk.onnx + tokenizer.json: each whole fine-tuned network
                    (encoder and head); only where its weights are (they stay on the machine that trained them)
M3's small network needs no export: without PyTorch it runs in NumPy (erp.models.mlp).

Nothing is approximated. The graphs compute in float32, as PyTorch does on the CPU; DistilBERT's weights stay at
the half precision they are stored in, each cast back to float32 inside the graph, exactly as DistilBertLearner.load
does. The command fails unless ONNX Runtime reproduces PyTorch's outputs within TOLERANCE, on texts from empty to
longer than the token limit. The service image runs it while it is built (see backend/Dockerfile):

  erp-export-onnx /service/models/arena-v1
"""

import argparse
import json
from pathlib import Path

import numpy as np

from erp.models import onnx_runtime
from erp.models.encoders import SbertEncoder

TOLERANCE = 1e-4  # as DistilBERT's reload check: batches pad differently and the CPU may round differently
OPSET = 18
CHECK_TEXTS = [
    "",
    "Fix typo",
    "As a product owner I want to export the backlog to CSV so that I can share it with stakeholders.",
    "NullPointerException in ScheduleManager.reload() when the cron expression is empty. ```java\nnull\n```",
    "Ünïcödé — façade naïve 日本語 emoji 🚀 and tabs\tand\nnew lines",
    " ".join(["The scheduler retries failed jobs with exponential backoff and records every attempt."] * 40),
]


def _export(module, args: tuple, path: Path, inputs: list[str], dynamic: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(module, args, str(path), input_names=inputs, output_names=["output"], dynamic_axes=dynamic,
                      opset_version=OPSET, dynamo=False)


def _store_half(path: Path) -> None:
    """Keep each large float32 weight that is exactly a half-precision number (DistilBERT's, which are stored in
    float16) as float16, with a Cast back to float32 in front of its use: the same numbers, half the file."""
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(path))
    graph, kept, casts = model.graph, [], []
    for weight in graph.initializer:
        values = numpy_helper.to_array(weight)
        if weight.data_type == onnx.TensorProto.FLOAT and values.size >= 1024:
            half = values.astype(np.float16)
            if np.array_equal(half.astype(np.float32), values):
                kept.append(numpy_helper.from_array(half, f"{weight.name}.float16"))
                casts.append(helper.make_node("Cast", [f"{weight.name}.float16"], [weight.name],
                                              to=onnx.TensorProto.FLOAT, name=f"{weight.name}.cast"))
                continue
        kept.append(weight)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    nodes = casts + list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _check(name: str, expected: np.ndarray, actual: np.ndarray) -> None:
    worst = float(np.max(np.abs(np.asarray(expected, np.float64) - np.asarray(actual, np.float64))))
    if not worst <= TOLERANCE:
        raise AssertionError(f"{name}: ONNX Runtime differs from PyTorch by {worst:g} (allowed {TOLERANCE:g})")
    print(f"{name}: ONNX Runtime matches PyTorch (largest difference {worst:.2g})")


def _save_tokenizer(hf, max_tokens: int, path: Path):
    """The Hugging Face tokenizer's own tokenizer.json, set to truncate and pad as the PyTorch code asks it to
    (truncation=True, max_length=max_tokens, padding=True)."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_str(hf.backend_tokenizer.to_str())
    tok.no_truncation()
    tok.no_padding()
    tok.enable_truncation(max_tokens)
    tok.enable_padding(pad_id=hf.pad_token_id, pad_token=hf.pad_token)
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    return tok


def export_sbert(directory: Path) -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(SbertEncoder.model_id, revision=SbertEncoder.revision, device="cpu")

    class Pooled(torch.nn.Module):
        """SBERT's Pooling (mean over the real tokens) and Normalize modules after the transformer."""

        def __init__(self):
            super().__init__()
            self.transformer = model[0].auto_model

        def forward(self, input_ids, attention_mask, token_type_ids):
            tokens = self.transformer(input_ids=input_ids, attention_mask=attention_mask,
                                      token_type_ids=token_type_ids).last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(tokens.size()).to(tokens.dtype)
            pooled = torch.sum(tokens * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
            return torch.nn.functional.normalize(pooled, p=2, dim=1)

    tok = _save_tokenizer(model.tokenizer, model.max_seq_length, directory / onnx_runtime.TOKENIZER)
    sample = onnx_runtime.tokenize(tok, CHECK_TEXTS[:2])
    names = ["input_ids", "attention_mask", "token_type_ids"]
    with torch.no_grad():
        _export(Pooled().eval(), tuple(torch.from_numpy(sample[n]) for n in names),
                directory / onnx_runtime.SbertModel.MODEL, names,
                {**{n: {0: "batch", 1: "tokens"} for n in names}, "output": {0: "batch"}})
    expected = model.encode(CHECK_TEXTS, batch_size=4, normalize_embeddings=True, show_progress_bar=False)
    exported = onnx_runtime.SbertModel(directory, SbertEncoder.dimensions)
    _check("sbert", expected, exported.encode(CHECK_TEXTS, batch_size=4))


def export_distilbert(source: Path, target: Path) -> None:
    import torch

    from erp.arena.distilbert import MAX_TOKENS, DistilBertLearner, _tokenizer

    manifest = json.loads((source / "model.json").read_text(encoding="utf-8"))
    tok = _save_tokenizer(_tokenizer(), MAX_TOKENS, target / onnx_runtime.TOKENIZER)
    rng = np.random.default_rng(42)
    for task in ("effort", "risk"):
        spec = manifest[task]["model"]
        learner = DistilBertLearner.load(source, task, spec, exported=False)
        features = rng.normal(size=(len(CHECK_TEXTS), spec["inputs"])).astype(np.float32)
        sample = onnx_runtime.tokenize(tok, CHECK_TEXTS[:2])
        path = target / Path(spec["file"]).with_suffix(".onnx").name
        with torch.no_grad():
            _export(learner.net.eval(), (torch.from_numpy(sample["input_ids"]),
                                         torch.from_numpy(sample["attention_mask"]),
                                         torch.from_numpy(features[:2])),
                    path, ["input_ids", "attention_mask", "features"],
                    {"input_ids": {0: "batch", 1: "tokens"}, "attention_mask": {0: "batch", 1: "tokens"},
                     "features": {0: "batch"}, "output": {0: "batch"}})
        _store_half(path)
        expected = learner._forward(CHECK_TEXTS, features, "cpu", size=4)
        net = onnx_runtime.DistilBertNet(path, target / onnx_runtime.TOKENIZER)
        _check(f"distilbert {task}", expected, net.forward(CHECK_TEXTS, features, size=4))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arena", type=Path, help="the arena folder (model.json files) to export into")
    args = parser.parse_args(argv)
    export_sbert(args.arena / "encoders" / "sbert")
    distilbert = args.arena / "distilbert"
    weights = [distilbert / f"{task}.safetensors" for task in ("effort", "risk")]
    if all(w.exists() for w in weights):
        export_distilbert(distilbert, distilbert)
    else:
        print("distilbert: no fine-tuned weights here, nothing to export (the service lists it as not installed)")


if __name__ == "__main__":
    main()
