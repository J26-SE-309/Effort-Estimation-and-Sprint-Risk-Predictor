import numpy as np
import pandas as pd
import pytest

from erp.features import catalog
from erp.models import first_models
from erp.models.bundle import ModelBundle
from erp.models.calibration import Calibrator, ConformalIntervals
from erp.models.encoders import SbertEncoder
from erp.models.inputs import design


def test_calibrators_round_trip_through_json():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 1, 2000)
    y = (rng.random(2000) < raw ** 2).astype(int)  # raw scores that overstate the risk
    for method in ("isotonic", "platt"):
        fitted = Calibrator(method).fit(raw, y)
        again = Calibrator.from_dict(fitted.to_dict())
        assert np.allclose(fitted.predict(raw), again.predict(raw))
        assert abs(fitted.predict([0.5])[0] - 0.25) < 0.08  # learnt that 0.5 really means about 0.25
    assert Calibrator("auto").fit(raw[:500], y[:500]).method == "platt"
    assert Calibrator("auto").fit(raw, y).method == "isotonic"


def test_conformal_quantile_and_interval():
    actual = np.log1p(np.array([1, 2, 3, 5, 8, 13, 3, 2, 5]))
    predicted = actual + np.linspace(-0.4, 0.4, 9)  # residuals 0.4, 0.3, ..., 0.4
    intervals = ConformalIntervals().fit(actual, predicted, coverages=(0.8,))
    # n = 9, k = ceil(10 * 0.8) = 8th smallest residual
    assert intervals.quantiles["0.8"] == pytest.approx(np.sort(np.abs(predicted - actual))[7])
    low, high = intervals.interval(np.log1p([3.0]), 0.8)
    assert low[0] < 3 < high[0] and low[0] >= 0
    assert ConformalIntervals.from_dict(intervals.to_dict()).quantiles == intervals.quantiles


def synthetic_stories(n: int = 600, seed: int = 1) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(index=pd.RangeIndex(1000, 1000 + n, name="Issue_ID"))
    for feature in catalog.FEATURES:
        if feature.name in ("issue_type", "priority_level", "project_key"):
            frame[feature.name] = rng.choice(["a", "b", "c"], n)
        elif feature.name.startswith(("has_", "invest_", "added_", "in_progress", "user_story", "mentions",
                                      "description_has")):
            frame[feature.name] = rng.random(n) < 0.3
        else:
            frame[feature.name] = rng.gamma(2.0, 2.0, n)
    frame["story_points"] = rng.choice([1, 2, 3, 5, 8], n)
    frame["at_risk"] = rng.random(n) < 0.5
    frame["split"] = np.where(np.arange(n) < 400, "train", np.where(np.arange(n) < 500, "cal", "test"))
    text = rng.normal(0, 1, (n, SbertEncoder.dimensions)).astype(np.float32)
    return frame, text


def test_bundle_saves_and_reloads_to_identical_predictions(tmp_path):
    frame, text = synthetic_stories()
    parts = frame["split"]
    m1 = first_models.fit("effort", design(frame, text, "effort"), np.log1p(frame["story_points"]), parts)
    m2 = first_models.fit("risk", design(frame, text, "risk"), frame["at_risk"].astype(int), parts)
    cal = (parts == "cal").to_numpy()
    raw = m2.predict_proba(design(frame, text, "risk"))[:, 1]
    bundle = ModelBundle(
        m1=m1.booster_, m2=m2.booster_,
        calibrator=Calibrator("platt").fit(raw[cal], frame["at_risk"][cal]),
        intervals=ConformalIntervals().fit(np.log1p(frame["story_points"][cal]),
                                           m1.predict(design(frame, text, "effort")[cal])),
        threshold=0.4, manifest={"name": "test"},
    )
    before = bundle.predict(frame, text)
    after = ModelBundle.load(bundle.save(tmp_path / "b")).predict(frame, text)
    pd.testing.assert_frame_equal(before, after)
    assert set(before["risk_level"]) <= {"low", "medium", "high"}
    assert (before["interval_0.8_low"] <= before["predicted_story_points"]).all()
    assert (before["predicted_story_points"] <= before["interval_0.8_high"]).all()
    assert "story_points" not in bundle.m1.feature_name()  # M1 never sees its own answer
