import numpy as np
import pandas as pd
import pytest

from erp.features import catalog
from erp.models import encoders, inputs, metrics, split

T = pd.Timestamp


def test_temporal_split_orders_each_projects_sprints_by_date():
    rows = [(project, sprint, T("2020-01-01") + pd.Timedelta(days=14 * sprint))
            for project in (1, 2) for sprint in range(10) for _ in range(3)]
    stories = pd.DataFrame(rows, columns=["Project_ID", "Sprint_ID", "sprint_start"]).sample(frac=1, random_state=0)
    parts = split.temporal_split(stories)
    by_sprint = stories.assign(part=parts).groupby(["Project_ID", "Sprint_ID"])["part"].agg(set)
    assert all(len(p) == 1 for p in by_sprint)  # a sprint never straddles two splits
    first = by_sprint.loc[1].map(lambda p: next(iter(p)))
    assert first.tolist() == ["train"] * 6 + ["cal"] * 2 + ["test"] * 2


def test_random_guessing_and_standardised_accuracy():
    y = [1, 3]  # the only other issue is always the guess: errors 2 and 2
    assert metrics.random_guessing_mae(y, [7, 7]) == 2.0
    assert metrics.standardised_accuracy(y, [1, 3], [7, 7]) == 100.0
    assert metrics.standardised_accuracy(y, [3, 1], [7, 7]) == 0.0
    # groups are never mixed: each issue is guessed from its own project
    assert metrics.random_guessing_mae([1, 1, 50, 50], ["a", "a", "b", "b"]) == 0.0


def test_expected_calibration_error():
    assert metrics.expected_calibration_error([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.0
    assert metrics.expected_calibration_error([0, 0], [0.9, 0.9]) == pytest.approx(0.9)


def test_threshold_keeps_the_recall_target():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.4, 0.3, 0.35, 0.2, 0.1, 0.1, 0.05, 0.05])
    threshold = metrics.choose_threshold(y, p, min_recall=0.75)
    report = metrics.classification_report(y, p, threshold)
    assert report["recall"] >= 0.75 and threshold == 0.3  # all four found, one false alarm: F1 0.89


class FakeModel:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, **_):
        self.calls += 1
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)


def test_sbert_encoder_caches_by_text(tmp_path):
    model = FakeModel()
    encoder = encoders.SbertEncoder(store=tmp_path / "e.npz", model=model)
    first = encoder.transform(pd.Series(["a", "bb", "a"]))
    again = encoders.SbertEncoder(store=tmp_path / "e.npz", model=model).transform(pd.Series(["bb", "a"]))
    assert first.tolist() == [[1, 1], [2, 1], [1, 1]] and again.tolist() == [[2, 1], [1, 1]]
    assert model.calls == 1  # the second encoder found everything in the cache


def test_effort_model_never_sees_story_points():
    frame = pd.DataFrame({name: [0] for name in catalog.names()})
    assert "story_points" not in inputs.structured(frame, "effort").columns
    assert "story_points" in inputs.structured(frame, "risk").columns
