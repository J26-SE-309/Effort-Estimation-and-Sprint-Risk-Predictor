"""E4: fine-tuned DistilBERT for effort (M1) and risk (M2), encoder and learner trained as one piece.

The story text goes through DistilBERT (distilbert-base-uncased, pinned revision, first 256 word pieces);
its first-token vector is joined with the structured features (TabularPrep) and a small head (768 + features
-> 256 -> 1) predicts standardised log(1 + story points) with a Huber loss, or the at-risk logit with binary
cross-entropy. Every layer of DistilBERT is retrained with the head.

Tuning: fine-tuning is far too costly for 25 Optuna trials, so it uses the grid the BERT authors recommend
(Devlin et al., 2019): learning rate 2e-5, 3e-5 or 5e-5 and batch size 16 or 32, up to 4 epochs with the best
epoch kept, scored on the same inner folds and with the same metrics as every other configuration.

Two steps, because fine-tuning needs a GPU while the platform predicts on the CPU (NFR12):
  erp-train-distilbert            GPU environment: tune, fine-tune on the training split (stopping on the
                                  calibration split), save the weights to ml-engine/models/arena-v1/distilbert/
  erp-train-distilbert --finish   CPU environment: predict every story on the CPU, fit C1, C2 and the threshold
                                  on the calibration split, write model.json and check the reload
The weights are stored at half precision (about 135 MB per task); the CPU predictions are made from exactly
those stored weights, so the reported numbers are the ones the service will reproduce.
"""

import argparse
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from erp.arena import configs
from erp.arena.data import MODELS_DIR, WORK_DIR, ArenaData, targets
from erp.arena.train import directory_size, fold_score, log, uncertainty
from erp.models import encoders
from erp.models.bundle import RISK_BANDS, code_version
from erp.models.inputs import TabularPrep, design

MODEL_ID = "distilbert/distilbert-base-uncased"
REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"  # the Hugging Face commit of the weights we start from
MAX_TOKENS = 256
GRID = [{"lr": lr, "batch_size": size} for lr in (2e-5, 3e-5, 5e-5) for size in (16, 32)]
MAX_EPOCHS = 4
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_SHARE = 0.1
DROPOUT = 0.1
HUBER_DELTA = 1.0
DIRECTORY = MODELS_DIR / "distilbert"
TRAINING = "training.json"


def _net(structured_inputs: int, pretrained: bool):
    import torch
    from torch import nn
    from transformers import AutoConfig, AutoModel

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = (AutoModel.from_pretrained(MODEL_ID, revision=REVISION) if pretrained
                         else AutoModel.from_config(AutoConfig.from_pretrained(MODEL_ID, revision=REVISION)))
            width = self.bert.config.dim
            self.head = nn.Sequential(nn.Linear(width + structured_inputs, 256), nn.ReLU(), nn.Dropout(DROPOUT),
                                      nn.Linear(256, 1))

        def forward(self, input_ids, attention_mask, features):
            first = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            return self.head(torch.cat([first, features], dim=1)).squeeze(-1)

    return Net()


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)


class DistilBertLearner:
    name, label = "distilbert", "Fine-tuned DistilBERT"

    def __init__(self, task: str):
        self.task = task
        self.prep: TabularPrep | None = None
        self.net = None
        self.params: dict = {}
        self.target: dict = {}
        self.pos_weight = 1.0
        self.best_epoch = 0
        self.best_val_loss = float("inf")

    def _batches(self, texts: list[str], features: np.ndarray, order, size: int, device):
        import torch

        tokenizer = self._tok = getattr(self, "_tok", None) or _tokenizer()
        for start in range(0, len(order), size):
            rows = order[start:start + size]
            encoded = tokenizer([texts[i] for i in rows], truncation=True, max_length=MAX_TOKENS, padding=True,
                                return_tensors="pt")
            yield rows, (encoded["input_ids"].to(device), encoded["attention_mask"].to(device),
                         torch.from_numpy(features[rows]).to(device))

    def _target(self, y) -> np.ndarray:
        y = np.asarray(y, float)
        return ((y - self.target["mean"]) / self.target["std"]) if self.task == "effort" else y

    def _loss(self, out, target):
        import torch
        from torch.nn import functional

        if self.task == "effort":
            return functional.huber_loss(out, target, delta=HUBER_DELTA)
        return functional.binary_cross_entropy_with_logits(out, target, pos_weight=torch.tensor(
            self.pos_weight, device=out.device))

    def fit(self, texts: pd.Series, x: pd.DataFrame, y, val_texts: pd.Series, x_val: pd.DataFrame, y_val,
            params: dict, seed: int = configs.SEED, device: str = "cuda") -> "DistilBertLearner":
        import torch
        from transformers import get_linear_schedule_with_warmup

        self.params = dict(params)
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        self.prep = TabularPrep(scale=True).fit(x)
        y = np.asarray(y, float)
        if self.task == "effort":
            self.target = {"mean": float(y.mean()), "std": float(y.std())}
        else:
            self.pos_weight = float((1 - y.mean()) / y.mean())
        features, val_features = self.prep.transform(x), self.prep.transform(x_val)
        target = torch.from_numpy(self._target(y).astype(np.float32))
        val_target = self._target(y_val).astype(np.float32)
        texts, val_texts = list(texts), list(val_texts)
        self.net = _net(features.shape[1], pretrained=True).to(device)
        optimiser = torch.optim.AdamW([
            {"params": self.net.bert.parameters(), "lr": params["lr"]},
            {"params": self.net.head.parameters(), "lr": HEAD_LR}], weight_decay=WEIGHT_DECAY)
        steps = MAX_EPOCHS * int(np.ceil(len(texts) / params["batch_size"]))
        schedule = get_linear_schedule_with_warmup(optimiser, int(WARMUP_SHARE * steps), steps)
        best_state = None
        for epoch in range(1, MAX_EPOCHS + 1):
            self.net.train()
            order = torch.randperm(len(texts), generator=generator).numpy()
            for rows, (ids, mask, feats) in self._batches(texts, features, order, params["batch_size"], device):
                optimiser.zero_grad()
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    out = self.net(ids, mask, feats)
                self._loss(out.float(), target[rows].to(device)).backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimiser.step()
                schedule.step()
            predicted = self._forward(val_texts, val_features, device)
            val_loss = float(self._loss(torch.from_numpy(predicted), torch.from_numpy(val_target)))
            log(f"    {self.task} epoch {epoch}: validation loss {val_loss:.4f}")
            if val_loss < self.best_val_loss:
                self.best_val_loss, self.best_epoch = val_loss, epoch
                best_state = {k: v.detach().to("cpu", copy=True) for k, v in self.net.state_dict().items()}
        self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def _forward(self, texts: list[str], features: np.ndarray, device: str, size: int = 64) -> np.ndarray:
        """Raw network outputs (standardised effort or risk logit)."""
        import torch

        self.net.eval()
        out = np.zeros(len(texts), dtype=np.float32)
        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            for rows, batch in self._batches(texts, features, np.arange(len(texts)), size, device):
                out[rows] = self.net(*batch).float().cpu().numpy()
        return out

    def predict(self, texts: pd.Series, x: pd.DataFrame, device: str = "cpu") -> np.ndarray:
        out = self._forward(list(texts), self.prep.transform(x), device).astype(float)
        if self.task == "effort":
            return out * self.target["std"] + self.target["mean"]
        return 1 / (1 + np.exp(-out))

    def save(self, directory: Path, prefix: str) -> dict:
        import torch
        from safetensors.torch import save_file

        directory.mkdir(parents=True, exist_ok=True)
        half = {k: (v.to(torch.float16) if v.is_floating_point() else v).contiguous().cpu()
                for k, v in self.net.state_dict().items()}
        save_file(half, directory / f"{prefix}.safetensors")
        return {"file": f"{prefix}.safetensors", "base_model": MODEL_ID, "revision": REVISION,
                "max_tokens": MAX_TOKENS, "stored_as": "float16", "params": self.params, "target": self.target,
                "pos_weight": self.pos_weight, "best_epoch": self.best_epoch, "best_val_loss": self.best_val_loss,
                "inputs": int(len(self.prep.columns())), "prep": self.prep.to_dict()}

    @classmethod
    def load(cls, directory: Path, task: str, spec: dict) -> "DistilBertLearner":
        """On the CPU, in float32, from the stored half-precision weights."""
        import torch
        from safetensors.torch import load_file

        learner = cls(task)
        learner.prep = TabularPrep.from_dict(spec["prep"])
        learner.params, learner.target, learner.pos_weight = spec["params"], spec["target"], spec["pos_weight"]
        learner.best_epoch, learner.best_val_loss = spec["best_epoch"], spec["best_val_loss"]
        learner.net = _net(spec["inputs"], pretrained=False)
        state = {k: (v.to(torch.float32) if v.is_floating_point() else v)
                 for k, v in load_file(directory / spec["file"]).items()}
        learner.net.load_state_dict(state)
        learner.net.eval()
        return learner


class DistilBertPredictor:
    """Same interface as arena.predictor.Predictor, for the router and the report."""

    encoder_name = "distilbert"

    def __init__(self, directory: Path):
        from erp.models.calibration import Calibrator, ConformalIntervals

        self.directory = directory
        self.manifest = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        self.levels = self.manifest["levels"]
        self.models = {t: DistilBertLearner.load(directory, t, self.manifest[t]["model"]) for t in ("effort", "risk")}
        self.intervals = ConformalIntervals.from_dict(self.manifest["effort"]["intervals"])
        self.calibrator = Calibrator.from_dict(self.manifest["risk"]["calibrator"])
        self.threshold, self.bands = self.manifest["risk"]["threshold"], self.manifest["risk"]["bands"]

    def encode(self, stories: pd.DataFrame, fresh: bool = False) -> pd.Series:
        return encoders.story_text(stories)  # DistilBERT reads the text itself

    def raw(self, stories: pd.DataFrame, text: pd.Series | None = None) -> tuple[np.ndarray, np.ndarray]:
        text = encoders.story_text(stories) if text is None else text
        out = {t: m.predict(text, design(stories, None, t, use_text=False, levels=self.levels))
               for t, m in self.models.items()}
        return out["effort"], out["risk"]

    def predict(self, stories: pd.DataFrame, text=None) -> pd.DataFrame:
        from erp.arena.predictor import finish

        log_points, raw = self.raw(stories, text)
        return finish(log_points, raw, self.intervals, self.calibrator, self.threshold, self.bands, stories.index)


def tune(arena: ArenaData, task: str) -> dict:
    """The grid on the inner folds; returns the tuning summary with the best settings."""
    train, texts = arena.train, arena.texts[arena.mask("train")]
    x = design(train, None, task, use_text=False, levels=arena.levels)
    y = targets(train, task)
    started, results = time.perf_counter(), []
    for params in GRID:
        scores, sizes = [], []
        for i, (fit, val) in enumerate(arena.folds):
            learner = DistilBertLearner(task).fit(texts[fit], x[fit], y[fit], texts[val], x[val], y[val], params)
            scores.append(fold_score(task, y[val], learner.predict(texts[val], x[val], device="cuda")))
            sizes.append(int(val.sum()))
            log(f"  {task} {params} fold {i + 1}: {scores[-1]:.4f}")
        results.append({"params": params, "value": float(np.average(scores, weights=sizes))})
    pick = min if task == "effort" else max
    best = pick(results, key=lambda r: r["value"])
    return {"trials": len(GRID), "complete": len(GRID), "pruned": 0,
            "metric": "MAE in story points" if task == "effort" else "ROC-AUC", "inner_folds": configs.INNER_FOLDS,
            "sampler": "grid of Devlin et al. (2019): learning rate x batch size", "grid": results,
            "best_value": best["value"], "best_params": best["params"],
            "seconds": round(time.perf_counter() - started, 1)}


def train_on_gpu(arena: ArenaData) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("Fine-tuning needs a CUDA GPU; run this in the GPU environment (see the README).")
    tr, cal = arena.mask("train"), arena.mask("cal")
    record_path = DIRECTORY / TRAINING
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
    for task in ("effort", "risk"):
        if task in record:
            log(f"{task}: already fine-tuned, skipped")
            continue
        log(f"{task}: tuning on the inner folds ({len(GRID)} settings x {configs.INNER_FOLDS} folds)")
        tuning = tune(arena, task)
        x = design(arena.frame, None, task, use_text=False, levels=arena.levels)
        y = targets(arena.frame, task)
        started = time.perf_counter()
        learner = DistilBertLearner(task).fit(arena.texts[tr], x[tr], y[tr], arena.texts[cal], x[cal], y[cal],
                                              tuning["best_params"])
        record[task] = {"model": learner.save(DIRECTORY, task), "inputs": list(x.columns), "tuning": tuning,
                        "fit_seconds": round(time.perf_counter() - started, 1),
                        "gpu": torch.cuda.get_device_name(0), "code": code_version()}
        record_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        log(f"{task}: fine-tuned (best epoch {learner.best_epoch}); saved to {DIRECTORY}")


def finish_on_cpu(arena: ArenaData) -> None:
    record = json.loads((DIRECTORY / TRAINING).read_text(encoding="utf-8"))
    outputs = {}
    for task in ("effort", "risk"):
        learner = DistilBertLearner.load(DIRECTORY, task, record[task]["model"])
        log(f"{task}: predicting {len(arena.frame):,} stories on the CPU")
        outputs[task] = learner.predict(arena.texts, design(arena.frame, None, task, use_text=False,
                                                            levels=arena.levels))
    fitted = uncertainty(arena, outputs["effort"], outputs["risk"])
    config = configs.BY_ID["distilbert"]
    manifest = {
        "config": asdict(config), "arena": configs.ARENA, "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code_version(),
        "encoder": {"name": "distilbert", "model_id": MODEL_ID, "revision": REVISION, "max_tokens": MAX_TOKENS,
                    "text": encoders.SbertEncoder.text.replace("256 word pieces", f"{MAX_TOKENS} word pieces")},
        "levels": arena.levels,
        "data": {"source": "TAWOS v1.1", "stories": len(arena.frame), "split": arena.parts.value_counts().to_dict()},
        "effort": {**record["effort"], "intervals": fitted["intervals"].to_dict()},
        "risk": {**record["risk"], "calibrator": fitted["calibrator"].to_dict(), "threshold": fitted["threshold"],
                 "bands": dict(RISK_BANDS)},
    }
    path = DIRECTORY / "model.json"
    path.write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    pd.DataFrame({"split": arena.parts, "effort_log": outputs["effort"], "risk_raw": outputs["risk"]},
                 index=arena.frame.index).to_parquet(WORK_DIR / "predictions" / "distilbert.parquet")
    test = arena.frame[arena.mask("test")]
    again = DistilBertPredictor(DIRECTORY).raw(test)
    # batches pad to different lengths, and the CPU's parallel matrix products may round differently
    tolerance = 1e-4
    same = all(np.allclose(a, outputs[t][arena.mask("test")], atol=tolerance, rtol=0)
               for a, t in zip(again, ("effort", "risk"), strict=True))
    if not same:
        raise AssertionError("distilbert: the reloaded model does not reproduce its predictions")
    manifest["check"] = {"round_trip": f"reloaded model reproduces the test predictions within {tolerance:g}",
                         "size_mb": directory_size(DIRECTORY)}
    path.write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    log(f"distilbert: model.json written ({manifest['check']['size_mb']} MB)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--finish", action="store_true", help="CPU step: predictions, C1, C2, model.json")
    args = parser.parse_args(argv)
    arena = ArenaData.load()
    (finish_on_cpu if args.finish else train_on_gpu)(arena)


if __name__ == "__main__":
    main()
