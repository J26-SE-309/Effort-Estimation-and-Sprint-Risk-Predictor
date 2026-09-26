"""A trained model bundle: everything needed to turn stories into the proposal's Appendix C prediction.

One folder per bundle version under ml-engine/models/, committed with the code:
  m1.txt, m2.txt   the LightGBM models in LightGBM's own text format (safe to open, stable across versions)
  bundle.json      the rest: encoder and its pinned revision, input columns, C1 calibrator, C2 interval
                   quantiles, risk threshold and bands, planning factors, test metrics, data and code versions
There are no pickles, so loading a bundle cannot run code, and a library upgrade cannot make it unreadable.
The FastAPI service will load a bundle at start-up (ML guide 8, step 10; Phase 5).
"""

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from erp import config
from erp.features import catalog
from erp.models import encoders, explain, inputs
from erp.models.calibration import Calibrator, ConformalIntervals

BUNDLE_FILE = "bundle.json"
RISK_BANDS = {"medium": 0.3, "high": 0.6}  # ML guide 4.2: < 0.3 low, 0.3-0.6 medium, > 0.6 high
INTERVAL_COVERAGES = (0.8, 0.9)


def code_version() -> dict:
    """The commit the bundle was trained with, and whether the working tree had uncommitted changes."""
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=config.REPO_ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""
    changed = bool(git("status", "--porcelain", "ml-engine/src"))
    return {"commit": git("rev-parse", "HEAD"), "uncommitted_changes": changed}


@dataclass
class ModelBundle:
    m1: object  # lightgbm.Booster
    m2: object
    calibrator: Calibrator
    intervals: ConformalIntervals
    threshold: float
    bands: dict = field(default_factory=lambda: dict(RISK_BANDS))
    manifest: dict = field(default_factory=dict)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        self.m1.save_model(directory / "m1.txt")
        self.m2.save_model(directory / "m2.txt")
        manifest = {
            **self.manifest,
            "m1": {"file": "m1.txt", "predicts": "log(1 + story points)", "inputs": self.m1.feature_name(),
                   "intervals": self.intervals.to_dict()},
            "m2": {"file": "m2.txt", "predicts": "raw at-risk score", "inputs": self.m2.feature_name(),
                   "calibrator": self.calibrator.to_dict(), "threshold": self.threshold, "bands": self.bands},
            "factors": {column: catalog.factor_of(column) for column in self.m2.feature_name()},
        }
        (directory / BUNDLE_FILE).write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: Path) -> "ModelBundle":
        import lightgbm as lgb

        manifest = json.loads((directory / BUNDLE_FILE).read_text(encoding="utf-8"))
        return cls(
            m1=lgb.Booster(model_file=str(directory / manifest["m1"]["file"])),
            m2=lgb.Booster(model_file=str(directory / manifest["m2"]["file"])),
            calibrator=Calibrator.from_dict(manifest["m2"]["calibrator"]),
            intervals=ConformalIntervals.from_dict(manifest["m1"]["intervals"]),
            threshold=manifest["m2"]["threshold"], bands=manifest["m2"]["bands"], manifest=manifest,
        )

    def predict(self, stories: pd.DataFrame, text: np.ndarray | None = None, explain_top: int = 3) -> pd.DataFrame:
        """Predictions for stories that have the catalogue features plus title and description_text.

        text: their SBERT vectors if already computed; otherwise they are encoded here.
        """
        if text is None:
            text = encoders.SbertEncoder().transform(encoders.story_text(stories))
        x1 = inputs.design(stories, text, "effort")[self.m1.feature_name()]
        x2 = inputs.design(stories, text, "risk")[self.m2.feature_name()]
        log_points = self.m1.predict(x1)
        probability = self.calibrator.predict(self.m2.predict(x2))
        out = pd.DataFrame({"predicted_story_points": np.expm1(log_points), "log_points": log_points,
                            "spillover_probability": probability}, index=stories.index)
        for coverage in INTERVAL_COVERAGES:
            low, high = self.intervals.interval(log_points, coverage)
            out[f"interval_{coverage:g}_low"], out[f"interval_{coverage:g}_high"] = low, high
        out["at_risk"] = probability >= self.threshold
        out["risk_level"] = np.select([probability >= self.bands["high"], probability >= self.bands["medium"]],
                                      ["high", "medium"], default="low")
        if explain_top:
            effort = explain.by_factor(explain.contributions(self.m1, x1), catalog.factor_of)
            risk = explain.by_factor(explain.contributions(self.m2, x2), catalog.factor_of)
            out["effort_reasons"] = explain.top_reasons(effort, explain_top)
            out["risk_reasons"] = explain.top_reasons(risk, explain_top)
        return out
