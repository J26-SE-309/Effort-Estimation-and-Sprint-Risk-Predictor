"""Load a saved arena configuration and turn stories into predictions (the router's building block, Phase 5).

Every configuration folder under ml-engine/models/arena-v1/ holds model.json plus its model files; encoders
fitted on our text are shared in encoders/. A predictor gives the same output for every configuration:
predicted story points with C2 intervals, the C1-calibrated risk probability with its level and flag, and
the confidence band (erp.models.confidence).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from erp.arena import data
from erp.models import encoders
from erp.models.bundle import INTERVAL_COVERAGES
from erp.models.calibration import Calibrator, load_intervals
from erp.models.confidence import Confidence
from erp.models.inputs import design
from erp.models.learners import LEARNERS
from erp.models.mlp import MLP

MANIFEST = "model.json"


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def with_estimated_points(stories: pd.DataFrame, log_points) -> pd.DataFrame:
    """M2 reads the story's committed points; a story without an estimate gets M1's (ML guide 4.2).

    Every training story had points, so this changes nothing there; it matters for new backlog items.
    """
    if "story_points" not in stories or not stories["story_points"].isna().any():
        return stories
    filled = stories.copy()
    missing = filled["story_points"].isna().to_numpy()
    filled.loc[missing, "story_points"] = np.expm1(np.asarray(log_points, float)[missing])
    return filled


@dataclass
class Run:
    """One pass through a configuration: its text vectors, model inputs and raw outputs."""

    text: object
    inputs: dict  # "effort" and "risk", or "joint" for M3
    log_points: np.ndarray
    raw: np.ndarray


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
        load_uncertainty(self)

    def encode(self, stories: pd.DataFrame, fresh: bool = False) -> np.ndarray:
        """fresh: encode even when a cached SBERT vector exists (what a never-seen story costs)."""
        texts = encoders.story_text(stories)
        if self.encoder_name == "sbert" and fresh:
            return self.encoder.encode(list(texts))
        return self.encoder.transform(texts)

    def run(self, stories: pd.DataFrame, text: np.ndarray | None = None) -> Run:
        text = self.encode(stories) if text is None else text
        columns = data.text_columns(self.encoder_name)
        if hasattr(self, "joint"):
            x = design(stories, text, "joint", text_columns=columns, levels=self.levels)
            out = self.joint.predict(x)
            return Run(text, {"joint": x}, out["effort"], out["risk"])
        x1 = design(stories, text, "effort", text_columns=columns, levels=self.levels)
        log_points = self.effort_model.predict(x1)
        x2 = design(with_estimated_points(stories, log_points), text, "risk", text_columns=columns,
                    levels=self.levels)
        return Run(text, {"effort": x1, "risk": x2}, log_points, self.risk_model.predict(x2))

    def raw(self, stories: pd.DataFrame, text: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(log(1 + points), risk score before C1)."""
        run = self.run(stories, text)
        return run.log_points, run.raw

    def predict(self, stories: pd.DataFrame, text: np.ndarray | None = None, missing_groups=0) -> pd.DataFrame:
        log_points, raw = self.raw(stories, text)
        return finish(self, stories, log_points, raw, missing_groups)


class StackPredictor:
    """The stacked ensemble: its base configurations' outputs combined by two small linear models."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
        bases = set(self.manifest["effort"]["bases"]) | set(self.manifest["risk"]["bases"])
        self.bases = {name: Predictor(directory.parent / name) for name in sorted(bases)}
        load_uncertainty(self)

    def raw(self, stories: pd.DataFrame, text: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
        outputs = {name: base.raw(stories, None if text is None else text.get(base.encoder_name))
                   for name, base in self.bases.items()}
        return combine(self.manifest, outputs)

    def predict(self, stories: pd.DataFrame, text: dict | None = None, missing_groups=0) -> pd.DataFrame:
        log_points, raw = self.raw(stories, text)
        return finish(self, stories, log_points, raw, missing_groups)


def combine(manifest: dict, outputs: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """The stack's meta-models: effort is a non-negative blend of the bases' log predictions; risk is a
    logistic regression on the bases' score logits."""
    effort, risk = manifest["effort"]["meta"], manifest["risk"]["meta"]
    log_points = effort["intercept"] + sum(w * outputs[name][0] for name, w in zip(
        manifest["effort"]["bases"], effort["weights"], strict=True))
    z = risk["intercept"] + sum(w * _logit(outputs[name][1]) for name, w in zip(
        manifest["risk"]["bases"], risk["weights"], strict=True))
    return np.asarray(log_points, float), 1 / (1 + np.exp(-z))


def load_uncertainty(model) -> None:
    """C2, C1, the threshold, the risk bands and the confidence score from model.json."""
    effort, risk = model.manifest["effort"], model.manifest["risk"]
    model.intervals = load_intervals(effort["intervals"], model.directory, model.manifest.get("levels"))
    model.calibrator = Calibrator.from_dict(risk["calibrator"])
    model.threshold, model.bands = risk["threshold"], risk["bands"]
    spec = model.manifest.get("confidence")
    model.confidence = Confidence.from_dict(spec) if spec else None


def finish(model, stories: pd.DataFrame, log_points, raw, missing_groups=0) -> pd.DataFrame:
    """From raw outputs to the prediction: points and C2 intervals, C1 probability, risk level, confidence.

    missing_groups: how many upstream feature groups were unavailable for each story (FR17), for confidence.
    """
    probability = model.calibrator.predict(raw)
    out = pd.DataFrame({"predicted_story_points": np.expm1(log_points), "log_points": log_points,
                        "raw_score": raw, "spillover_probability": probability}, index=stories.index)
    for coverage in INTERVAL_COVERAGES:
        low, high = model.intervals.bounds(stories, log_points, coverage)
        out[f"interval_{coverage:g}_low"], out[f"interval_{coverage:g}_high"] = low, high
    out["at_risk"] = probability >= model.threshold
    out["risk_level"] = np.select([probability >= model.bands["high"], probability >= model.bands["medium"]],
                                  ["high", "medium"], default="low")
    if model.confidence is not None:
        out = out.join(model.confidence.score(stories, out["interval_0.8_low"], out["interval_0.8_high"],
                                              out["predicted_story_points"], probability, missing_groups))
    return out


def load(directory: Path):
    learner = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))["config"]["learner"]
    if learner == "stack":
        return StackPredictor(directory)
    if learner == "distilbert":
        from erp.arena.distilbert import DistilBertPredictor

        return DistilBertPredictor(directory)
    return Predictor(directory)
