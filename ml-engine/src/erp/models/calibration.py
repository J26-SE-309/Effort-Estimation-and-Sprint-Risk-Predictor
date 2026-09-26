"""Uncertainty models fitted on the calibration split: C1 for risk probabilities, C2 for effort intervals.

Both are a handful of numbers, so they are saved as JSON inside the model bundle (no pickles) and rebuilt
exactly from it.
"""

import math

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ISOTONIC_MIN_SAMPLES = 1000  # ML guide 4.5: isotonic can overfit below about a thousand calibration samples


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Calibrator:
    """C1: maps M2's raw scores onto probabilities that match how often stories really were at risk.

    method 'isotonic' (a monotone step curve, for large calibration sets) or 'platt' (a logistic curve on the
    log-odds of the raw score). 'auto' follows the ML guide: isotonic from ISOTONIC_MIN_SAMPLES samples on.
    """

    def __init__(self, method: str = "auto"):
        self.method = method
        self.params: dict = {}

    def fit(self, raw_scores, y_true) -> "Calibrator":
        raw, y = np.asarray(raw_scores, float), np.asarray(y_true, int)
        if self.method == "auto":
            self.method = "isotonic" if len(raw) >= ISOTONIC_MIN_SAMPLES else "platt"
        if self.method == "isotonic":
            curve = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, y)
            self.params = {"x": curve.X_thresholds_.round(6).tolist(), "y": curve.y_thresholds_.round(6).tolist()}
        elif self.method == "platt":
            model = LogisticRegression(C=1e6).fit(_logit(raw).reshape(-1, 1), y)
            self.params = {"a": float(model.coef_[0, 0]), "b": float(model.intercept_[0])}
        else:
            raise ValueError(f"unknown calibration method {self.method!r}")
        return self

    def predict(self, raw_scores) -> np.ndarray:
        raw = np.asarray(raw_scores, float)
        if self.method == "isotonic":
            return np.interp(raw, self.params["x"], self.params["y"])
        z = self.params["a"] * _logit(raw) + self.params["b"]
        return 1 / (1 + np.exp(-z))

    def to_dict(self) -> dict:
        return {"method": self.method, **self.params}

    @classmethod
    def from_dict(cls, data: dict) -> "Calibrator":
        calibrator = cls(data["method"])
        calibrator.params = {k: v for k, v in data.items() if k != "method"}
        return calibrator


class ConformalIntervals:
    """C2: split conformal prediction intervals for M1, on the log(1 + story points) scale M1 predicts in.

    On the calibration split the absolute residuals |log1p(actual) - prediction| are sorted and, for coverage
    c, q is the ceil((n + 1) c)-th smallest. The interval [prediction - q, prediction + q] is turned back into
    story points, so it is multiplicative: wider for big stories, narrower for small ones. The guarantee holds
    if future stories resemble the calibration ones (ML guide 4.6); the test split checks it.
    """

    def __init__(self, quantiles: dict[str, float] | None = None):
        self.quantiles = quantiles or {}

    def fit(self, log_actual, log_predicted, coverages=(0.8, 0.9)) -> "ConformalIntervals":
        residuals = np.sort(np.abs(np.asarray(log_actual, float) - np.asarray(log_predicted, float)))
        n = len(residuals)
        for coverage in coverages:
            k = math.ceil((n + 1) * coverage) - 1
            self.quantiles[f"{coverage:g}"] = float(residuals[min(k, n - 1)])
        return self

    def interval(self, log_predicted, coverage: float) -> tuple[np.ndarray, np.ndarray]:
        q = self.quantiles[f"{coverage:g}"]
        centre = np.asarray(log_predicted, float)
        return np.maximum(np.expm1(centre - q), 0.0), np.expm1(centre + q)

    def to_dict(self) -> dict:
        return {"method": "split conformal on log(1 + story points)", "quantiles": self.quantiles}

    @classmethod
    def from_dict(cls, data: dict) -> "ConformalIntervals":
        return cls(dict(data["quantiles"]))


def coverage_and_width(actual, low, high) -> tuple[float, float]:
    actual = np.asarray(actual, float)
    inside = (actual >= low) & (actual <= high)
    return float(inside.mean()), float(np.mean(np.asarray(high) - np.asarray(low)))
