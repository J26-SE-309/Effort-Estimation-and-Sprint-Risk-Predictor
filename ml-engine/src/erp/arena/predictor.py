"""Load a saved arena configuration and turn stories into predictions (the router's building block, Phase 5).

Every configuration folder under ml-engine/models/arena-v1/ holds model.json plus its model files; encoders
fitted on our text are shared in encoders/. A predictor gives the same output for every configuration:
predicted story points with C2 intervals, and the C1-calibrated risk probability with its level and flag.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from erp.arena import data
from erp.models import encoders
from erp.models.bundle import INTERVAL_COVERAGES
from erp.models.calibration import Calibrator, ConformalIntervals
from erp.models.inputs import design
from erp.models.learners import LEARNERS
from erp.models.mlp import MLP

MANIFEST = "model.json"


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Predictor:
    """A single-task pair (one learner for effort, one for risk) or the joint network M3."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
        self.encoder_name = self.manifest["encoder"]["name"]
        self.encoder = data.load_encoder(self.encoder_name)
        self.levels = self.manifest["levels"]
        effort, risk = self.manifest["effort"], self.manifest["risk"]
        if "joint" in self.manifest:
            self.joint = MLP.load(directory, self.manifest["joint"])
        else:
            learner = LEARNERS[self.manifest["config"]["learner"]]
            self.effort_model = learner.load(directory, "effort", effort["model"])
            self.risk_model = learner.load(directory, "risk", risk["model"])
        self.intervals = ConformalIntervals.from_dict(effort["intervals"])
        self.calibrator = Calibrator.from_dict(risk["calibrator"])
        self.threshold, self.bands = risk["threshold"], risk["bands"]

    def encode(self, stories: pd.DataFrame, fresh: bool = False) -> np.ndarray:
        """fresh: encode even when a cached SBERT vector exists (what a never-seen story costs)."""
        texts = encoders.story_text(stories)
        if self.encoder_name == "sbert" and fresh:
            return self.encoder.encode(list(texts))
        return self.encoder.transform(texts)

    def raw(self, stories: pd.DataFrame, text: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(log(1 + points), risk score before C1)."""
        text = self.encode(stories) if text is None else text
        columns = data.text_columns(self.encoder_name)
        if hasattr(self, "joint"):
            out = self.joint.predict(design(stories, text, "joint", text_columns=columns, levels=self.levels))
            return out["effort"], out["risk"]
        x1 = design(stories, text, "effort", text_columns=columns, levels=self.levels)
        x2 = design(stories, text, "risk", text_columns=columns, levels=self.levels)
        return self.effort_model.predict(x1), self.risk_model.predict(x2)

    def predict(self, stories: pd.DataFrame, text: np.ndarray | None = None) -> pd.DataFrame:
        log_points, raw = self.raw(stories, text)
        return finish(log_points, raw, self.intervals, self.calibrator, self.threshold, self.bands, stories.index)


class StackPredictor:
    """The stacked ensemble: its base configurations' outputs combined by two small linear models."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
        bases = set(self.manifest["effort"]["bases"]) | set(self.manifest["risk"]["bases"])
        self.bases = {name: Predictor(directory.parent / name) for name in sorted(bases)}
        effort, risk = self.manifest["effort"], self.manifest["risk"]
        self.intervals = ConformalIntervals.from_dict(effort["intervals"])
        self.calibrator = Calibrator.from_dict(risk["calibrator"])
        self.threshold, self.bands = risk["threshold"], risk["bands"]

    def raw(self, stories: pd.DataFrame, text: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
        outputs = {name: base.raw(stories, None if text is None else text.get(base.encoder_name))
                   for name, base in self.bases.items()}
        return combine(self.manifest, outputs)

    def predict(self, stories: pd.DataFrame, text: dict | None = None) -> pd.DataFrame:
        log_points, raw = self.raw(stories, text)
        return finish(log_points, raw, self.intervals, self.calibrator, self.threshold, self.bands, stories.index)


def combine(manifest: dict, outputs: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """The stack's meta-models: effort is a non-negative blend of the bases' log predictions; risk is a
    logistic regression on the bases' score logits."""
    effort, risk = manifest["effort"]["meta"], manifest["risk"]["meta"]
    log_points = effort["intercept"] + sum(w * outputs[name][0] for name, w in zip(
        manifest["effort"]["bases"], effort["weights"], strict=True))
    z = risk["intercept"] + sum(w * _logit(outputs[name][1]) for name, w in zip(
        manifest["risk"]["bases"], risk["weights"], strict=True))
    return np.asarray(log_points, float), 1 / (1 + np.exp(-z))


def finish(log_points, raw, intervals, calibrator, threshold, bands, index) -> pd.DataFrame:
    probability = calibrator.predict(raw)
    out = pd.DataFrame({"predicted_story_points": np.expm1(log_points), "log_points": log_points,
                        "raw_score": raw, "spillover_probability": probability}, index=index)
    for coverage in INTERVAL_COVERAGES:
        low, high = intervals.interval(log_points, coverage)
        out[f"interval_{coverage:g}_low"], out[f"interval_{coverage:g}_high"] = low, high
    out["at_risk"] = probability >= threshold
    out["risk_level"] = np.select([probability >= bands["high"], probability >= bands["medium"]],
                                  ["high", "medium"], default="low")
    return out


def load(directory: Path):
    learner = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))["config"]["learner"]
    if learner == "stack":
        return StackPredictor(directory)
    if learner == "distilbert":
        from erp.arena.distilbert import DistilBertPredictor

        return DistilBertPredictor(directory)
    return Predictor(directory)
