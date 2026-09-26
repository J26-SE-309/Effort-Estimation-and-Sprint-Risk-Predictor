"""confidence_score: how much to trust one prediction, shown as High / Medium / Low (ML guide 4.6).

A proposal to agree with the supervisor; every number below is a documented starting value.

    confidence_score = S x (E + R) / 2

E  effort certainty: the share of calibration stories whose 80% interval was relatively wider than this one's
   (relative width = interval width / (predicted points + 1)). 0.8 = 'tighter than 80% of past estimates'.
   It needs adaptive intervals (C2): with plain split conformal every story has the same relative width.
R  risk certainty: |2p - 1| for the calibrated probability p. 0.5 is a coin flip (0), 0.1 or 0.9 a firm call.
S  data support, 1 unless the service knows the inputs are weak: x 0.7 for a team with fewer than 3 closed
   sprints (cold start), x 0.7 for a project the model never saw, x 0.85 for each upstream feature group
   (Components 1-3) that was unavailable (FR17).

Bands, never the raw number, go on the dashboard: High from 0.6, Medium from 0.35, Low below. The score is only
honest if the bands really order the accuracy (High more accurate than Medium, Medium than Low); erp-fit-intervals
checks that on the test split.
"""

import numpy as np
import pandas as pd

BANDS = {"high": 0.6, "medium": 0.35}
COLD_START_SPRINTS = 3
PENALTIES = {"cold_start": 0.7, "unseen_project": 0.7, "missing_group": 0.85}
REFERENCE_POINTS = 101  # percentiles of the calibration split's relative widths kept in model.json


def relative_width(low, high, predicted_points) -> np.ndarray:
    return (np.asarray(high, float) - np.asarray(low, float)) / (np.asarray(predicted_points, float) + 1)


class Confidence:
    def __init__(self, reference: list[float] | None = None, known_projects: list[str] | None = None):
        self.reference = reference or []
        self.known_projects = known_projects or []

    def fit(self, relative_widths, known_projects) -> "Confidence":
        percentiles = np.percentile(np.asarray(relative_widths, float), np.linspace(0, 100, REFERENCE_POINTS))
        self.reference = [round(float(v), 6) for v in percentiles]
        self.known_projects = sorted(map(str, known_projects))
        return self

    def effort_certainty(self, widths) -> np.ndarray:
        reference = np.asarray(self.reference)
        if reference[-1] - reference[0] < 1e-9:  # constant widths (split conformal): no information
            return np.full(len(np.atleast_1d(widths)), 0.5)
        share_narrower = np.interp(np.asarray(widths, float), reference, np.linspace(0, 1, len(reference)))
        return 1 - share_narrower

    @staticmethod
    def risk_certainty(probability) -> np.ndarray:
        return np.abs(2 * np.asarray(probability, float) - 1)

    def support(self, stories: pd.DataFrame, missing_groups=0) -> np.ndarray:
        support = np.ones(len(stories))
        support *= np.where(stories["history_sprints"].fillna(0).to_numpy() < COLD_START_SPRINTS,
                            PENALTIES["cold_start"], 1.0)
        support *= np.where(stories["project_key"].astype(str).isin(self.known_projects).to_numpy(), 1.0,
                            PENALTIES["unseen_project"])
        return support * PENALTIES["missing_group"] ** np.asarray(missing_groups, float)

    def score(self, stories: pd.DataFrame, low, high, predicted_points, probability,
              missing_groups=0) -> pd.DataFrame:
        effort = self.effort_certainty(relative_width(low, high, predicted_points))
        risk = self.risk_certainty(probability)
        support = self.support(stories, missing_groups)
        score = support * (effort + risk) / 2
        band = np.select([score >= BANDS["high"], score >= BANDS["medium"]], ["high", "medium"], default="low")
        return pd.DataFrame({"effort_certainty": effort, "risk_certainty": risk, "data_support": support,
                             "confidence_score": score, "confidence": band}, index=stories.index)

    def to_dict(self) -> dict:
        return {"formula": "S x (E + R) / 2", "bands": BANDS, "penalties": PENALTIES,
                "cold_start_sprints": COLD_START_SPRINTS, "reference_relative_widths": self.reference,
                "known_projects": self.known_projects}

    @classmethod
    def from_dict(cls, data: dict) -> "Confidence":
        return cls(data["reference_relative_widths"], data["known_projects"])
