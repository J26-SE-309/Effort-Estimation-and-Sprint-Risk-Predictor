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

    def bounds(self, stories, log_points, coverage: float) -> tuple[np.ndarray, np.ndarray]:
        """Same signature as AdaptiveIntervals.bounds; the stories are not needed here."""
        return self.interval(log_points, coverage)

    def to_dict(self) -> dict:
        return {"method": "split conformal on log(1 + story points)", "quantiles": self.quantiles}

    @classmethod
    def from_dict(cls, data: dict) -> "ConformalIntervals":
        return cls(dict(data["quantiles"]))


def coverage_and_width(actual, low, high) -> tuple[float, float]:
    actual = np.asarray(actual, float)
    inside = (actual >= low) & (actual <= high)
    return float(inside.mean()), float(np.mean(np.asarray(high) - np.asarray(low)))


class AdaptiveIntervals:
    """C2, adaptive: normalized split conformal on log(1 + story points) (Papadopoulos et al., 2008; Lei et al.,
    2018).

    Plain split conformal gives every story the same interval on the log scale, so every story's range is the
    same multiple of its prediction. Here a small LightGBM 'difficulty model' predicts how far off the effort
    model tends to be for a story like this one (from its features and its predicted size). It learns from
    the effort model's inner-fold predictions on training stories, each made by a model that never saw that
    story, never from the calibration split. On the calibration split the scores |actual - prediction| /
    difficulty are sorted and, for coverage c, q is the ceil((n + 1) c)-th smallest; the interval is
    prediction +- q x difficulty. The coverage guarantee is the same as before; the width now follows the story.
    """

    method = "normalized split conformal on log(1 + story points)"
    DIFFICULTY = {"objective": "l2", "n_estimators": 400, "learning_rate": 0.03, "num_leaves": 15,
                  "min_child_samples": 100, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
                  "random_state": 42, "verbose": -1}
    FILE = "effort-difficulty.txt"

    def __init__(self, levels: dict | None = None):
        self.levels = levels
        self.booster = None
        self.floor = 0.0
        self.quantiles: dict[str, float] = {}

    def inputs(self, stories, log_points):
        from erp.models.inputs import structured

        return structured(stories, "effort", self.levels).assign(predicted_log=np.asarray(log_points, float))

    def fit_difficulty(self, stories, log_points, log_actual) -> "AdaptiveIntervals":
        import lightgbm as lgb

        residuals = np.abs(np.asarray(log_actual, float) - np.asarray(log_points, float))
        model = lgb.LGBMRegressor(**self.DIFFICULTY).fit(self.inputs(stories, log_points), residuals)
        self.booster = model.booster_
        # a floor, so a story the model thinks is trivially easy cannot get a zero-width interval
        self.floor = float(np.percentile(self.booster.predict(self.inputs(stories, log_points)), 5))
        return self

    def difficulty(self, stories, log_points) -> np.ndarray:
        return np.maximum(self.booster.predict(self.inputs(stories, log_points)), self.floor)

    def fit(self, stories, log_actual, log_points, coverages=(0.8, 0.9)) -> "AdaptiveIntervals":
        scores = np.sort(np.abs(np.asarray(log_actual, float) - np.asarray(log_points, float))
                         / self.difficulty(stories, log_points))
        n = len(scores)
        for coverage in coverages:
            self.quantiles[f"{coverage:g}"] = float(scores[min(math.ceil((n + 1) * coverage) - 1, n - 1)])
        return self

    def bounds(self, stories, log_points, coverage: float) -> tuple[np.ndarray, np.ndarray]:
        centre = np.asarray(log_points, float)
        half = self.quantiles[f"{coverage:g}"] * self.difficulty(stories, centre)
        return np.maximum(np.expm1(centre - half), 0.0), np.expm1(centre + half)

    def save(self, directory) -> dict:
        self.booster.save_model(directory / self.FILE)
        return {"method": self.method, "difficulty_model": self.FILE, "difficulty_params": self.DIFFICULTY,
                "floor": self.floor, "quantiles": self.quantiles}

    @classmethod
    def load(cls, directory, spec: dict, levels: dict) -> "AdaptiveIntervals":
        import lightgbm as lgb

        intervals = cls(levels)
        intervals.booster = lgb.Booster(model_file=str(directory / spec["difficulty_model"]))
        intervals.floor, intervals.quantiles = spec["floor"], dict(spec["quantiles"])
        return intervals


def load_intervals(spec: dict, directory=None, levels: dict | None = None):
    """C2 from a model manifest: adaptive when it has a difficulty model, plain split conformal otherwise."""
    if spec.get("method") == AdaptiveIntervals.method:
        return AdaptiveIntervals.load(directory, spec, levels)
    return ConformalIntervals.from_dict(spec)
