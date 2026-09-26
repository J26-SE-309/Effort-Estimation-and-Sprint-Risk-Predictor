"""Adaptive C2 intervals and the confidence score."""

import numpy as np
import pandas as pd
from test_bundle import synthetic_stories

from erp.models import confidence as conf
from erp.models.calibration import AdaptiveIntervals, ConformalIntervals, load_intervals
from erp.models.inputs import category_levels


def test_adaptive_intervals_follow_difficulty_keep_coverage_and_reload(tmp_path):
    frame, _ = synthetic_stories(n=3000, seed=4)
    rng = np.random.default_rng(4)
    hard = (frame["issue_type"] == "a").to_numpy()  # one kind of story is three times harder to estimate
    truth = np.log1p(frame["story_points"].to_numpy(float))
    predicted = truth + rng.normal(0, np.where(hard, 0.6, 0.2))
    fit_rows, cal, test = np.arange(3000) < 1500, (np.arange(3000) >= 1500) & (np.arange(3000) < 2250), \
        np.arange(3000) >= 2250
    intervals = AdaptiveIntervals(category_levels(frame)).fit_difficulty(frame[fit_rows], predicted[fit_rows],
                                                                        truth[fit_rows])
    intervals.fit(frame[cal], truth[cal], predicted[cal])
    low, high = intervals.bounds(frame[test], predicted[test], 0.8)
    points = frame["story_points"].to_numpy(float)[test]
    inside = (points >= low) & (points <= high)
    assert abs(inside.mean() - 0.8) < 0.05
    width = np.log1p(high) - np.log1p(low)
    assert width[hard[test]].mean() > 2 * width[~hard[test]].mean()  # wider for the hard kind
    # plain split conformal covers hard stories far less often than easy ones; adaptive closes most of the gap
    split = ConformalIntervals().fit(truth[cal], predicted[cal], (0.8,))
    split_low, split_high = split.interval(predicted[test], 0.8)
    split_inside = (points >= split_low) & (points <= split_high)
    gap = lambda covered: abs(covered[hard[test]].mean() - covered[~hard[test]].mean())  # noqa: E731
    assert gap(inside) < gap(split_inside) / 3
    again = load_intervals(intervals.save(tmp_path), tmp_path, category_levels(frame))
    assert all(np.array_equal(a, b) for a, b in zip(again.bounds(frame[test], predicted[test], 0.8), (low, high),
                                                     strict=True))


def test_confidence_score_parts_penalties_and_bands():
    stories = pd.DataFrame({"history_sprints": [10, 10, 1, 10], "project_key": ["A", "A", "A", "NEW"]})
    confidence = conf.Confidence().fit(np.linspace(1.0, 2.0, 200), ["A"])
    points = np.array([3.0, 3.0, 3.0, 3.0])
    width = np.array([1.0, 2.0, 1.0, 1.0]) * (points + 1)  # relative widths 1 (tightest) and 2 (widest)
    out = confidence.score(stories, np.zeros(4), width, points, np.array([0.95, 0.5, 0.95, 0.95]))
    assert out["effort_certainty"].round(3).tolist() == [1.0, 0.0, 1.0, 1.0]
    assert out["risk_certainty"].round(3).tolist() == [0.9, 0.0, 0.9, 0.9]
    assert out["data_support"].tolist() == [1.0, 1.0, 0.7, 0.7]  # cold start, unseen project
    assert out["confidence"].tolist() == ["high", "low", "high", "high"]
    assert out["confidence_score"].round(3).tolist() == [0.95, 0.0, 0.665, 0.665]
    missing = confidence.score(stories.iloc[:1], np.zeros(1), width[:1], points[:1], np.array([0.95]),
                               missing_groups=3)
    assert missing["data_support"].iloc[0] == 0.85 ** 3 and missing["confidence"].iloc[0] == "medium"
    again = conf.Confidence.from_dict(confidence.to_dict())
    assert again.reference == confidence.reference and again.known_projects == ["A"]
    # plain split conformal: every relative width is the same, so E carries no information
    flat = conf.Confidence().fit(np.full(50, 1.36), ["A"])
    assert flat.effort_certainty(np.array([1.36, 1.36])).tolist() == [0.5, 0.5]


def test_tempering_power_is_saved_and_reloaded(tmp_path):
    frame, _ = synthetic_stories(n=800, seed=5)
    truth = np.log1p(frame["story_points"].to_numpy(float))
    predicted = truth + np.random.default_rng(5).normal(0, 0.3, 800)
    intervals = AdaptiveIntervals(category_levels(frame)).fit_difficulty(frame[:400], predicted[:400], truth[:400])
    raw = intervals.difficulty(frame[400:], predicted[400:])
    intervals.power = 0.5
    assert np.allclose(intervals.difficulty(frame[400:], predicted[400:]), raw ** 0.5)
    intervals.fit(frame[400:600], truth[400:600], predicted[400:600])
    again = load_intervals(intervals.save(tmp_path), tmp_path, category_levels(frame))
    assert again.power == 0.5
    assert all(np.array_equal(a, b) for a, b in zip(again.bounds(frame[600:], predicted[600:], 0.9),
                                                     intervals.bounds(frame[600:], predicted[600:], 0.9), strict=True))
