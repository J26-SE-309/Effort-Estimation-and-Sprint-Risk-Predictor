"""The arena's building blocks: time-ordered inner folds, encoders, learners, TabularPrep, M3 and the statistics."""

import numpy as np
import pandas as pd
import pytest
from test_bundle import synthetic_stories

from erp.arena.data import inner_folds
from erp.features import catalog
from erp.models import metrics
from erp.models.inputs import TabularPrep, category_levels, design
from erp.models.learners import LEARNERS


def test_inner_folds_only_look_forward_in_time():
    rows = []
    for project, sprints in (("A", 12), ("B", 5)):
        for sprint in range(sprints):
            for _ in range(3):
                rows.append({"Project_ID": project, "Sprint_ID": f"{project}{sprint}",
                             "sprint_start": pd.Timestamp("2020-01-01") + pd.Timedelta(days=14 * sprint)})
    train = pd.DataFrame(rows)
    folds = inner_folds(train, k=3)
    assert len(folds) == 3
    for fit, val in folds:
        assert not (fit & val).any() and val.any()
        for project in ("A", "B"):
            mine = (train["Project_ID"] == project).to_numpy()
            if (fit & mine).any() and (val & mine).any():
                assert train["sprint_start"][fit & mine].max() < train["sprint_start"][val & mine].min()
    # every sprint of a project stays on one side, and later folds train on more
    assert folds[0][0].sum() < folds[1][0].sum() < folds[2][0].sum()


def test_text_encoders_reload_to_identical_vectors(tmp_path, monkeypatch):
    pytest.importorskip("gensim")
    from erp.models.encoders import FastTextEncoder, TfidfEncoder

    monkeypatch.setattr(TfidfEncoder, "params", {**TfidfEncoder.params, "svd": 5, "min_df": 1})
    monkeypatch.setattr(TfidfEncoder, "dimensions", 5)
    rng = np.random.default_rng(0)
    words = ["login", "bug", "fix", "user", "page", "error", "api", "test", "slow", "crash"]
    texts = pd.Series([" ".join(rng.choice(words, 10)) for _ in range(200)])
    unseen = pd.Series(["login bug", "logn buug zzz", ""])
    for encoder_class in (TfidfEncoder, FastTextEncoder):
        encoder = encoder_class().fit(texts)
        encoder.save(tmp_path / encoder_class.name)
        again = encoder_class.load(tmp_path / encoder_class.name)
        assert np.array_equal(encoder.transform(unseen), again.transform(unseen))
        assert encoder.transform(unseen).shape == (3, encoder_class.dimensions)
    # FastText builds a misspelt word from its character pieces; an empty story is all zeros
    vectors = FastTextEncoder().fit(texts).transform(unseen)
    assert np.abs(vectors[1]).sum() > 0 and np.abs(vectors[2]).sum() == 0


PARAMS = {
    "lightgbm": {"learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 20, "colsample_bytree": 0.5,
                 "subsample": 0.8, "reg_lambda": 1.0},
    "xgboost": {"learning_rate": 0.05, "max_depth": 4, "min_child_weight": 1, "subsample": 0.8,
                "colsample_bytree": 0.5, "reg_lambda": 1, "reg_alpha": 0.01},
    "catboost": {"learning_rate": 0.1, "depth": 4, "l2_leaf_reg": 3, "random_strength": 1,
                 "bagging_temperature": 0.5, "rsm": 0.5},
    "random_forest": {"max_features": 0.3, "min_samples_leaf": 5, "max_samples": 0.8},
    "svm": {"C": 1.0, "gamma": 0.003},
}
NEEDS = {"xgboost": "xgboost", "catboost": "catboost", "random_forest": "skops", "svm": "skops"}


@pytest.mark.parametrize("task", ["effort", "risk"])
@pytest.mark.parametrize("name", list(LEARNERS))
def test_learners_reload_to_identical_predictions(name, task, tmp_path):
    if name in NEEDS:
        pytest.importorskip(NEEDS[name])
    frame, text = synthetic_stories()
    train, cal = (frame["split"] == "train").to_numpy(), (frame["split"] == "cal").to_numpy()
    x = design(frame, text[:, :10], task, text_columns=[f"sbert_{i}" for i in range(10)],
               levels=category_levels(frame))
    y = np.log1p(frame["story_points"]) if task == "effort" else frame["at_risk"].astype(int)
    params = dict(PARAMS[name], **({"epsilon": 0.1} if name == "svm" and task == "effort" else {}))
    learner = LEARNERS[name](task).fit(x[train], y[train], x[cal], y[cal], params)
    predicted = learner.predict(x[~train])
    again = LEARNERS[name].load(tmp_path, task, learner.save(tmp_path, task)).predict(x[~train])
    assert np.array_equal(predicted, again)
    if task == "risk":
        assert ((predicted > 0) & (predicted < 1)).all()
    assert "story_points" not in x.columns or task == "risk"


def test_tabular_prep_one_hot_missing_flags_and_json():
    x = pd.DataFrame({"a": [1.0, np.nan, 3.0], "project_key": pd.Categorical(["P", "Q", "P"])})
    prep = TabularPrep(scale=False).fit(x)
    assert prep.columns() == ["a", "a__missing", "project_key=P", "project_key=Q"]
    later = pd.DataFrame({"a": [np.nan], "project_key": pd.Categorical(["NEW"])})
    assert prep.transform(later).tolist() == [[2.0, 1.0, 0.0, 0.0]]  # median filled, flagged, unknown project
    again = TabularPrep.from_dict(prep.to_dict())
    assert np.array_equal(again.transform(x), prep.transform(x))
    assert catalog.factor_of("project_key=P") == catalog.factor_of("project_key")
    assert catalog.factor_of("reopen_rate__missing") == catalog.factor_of("reopen_rate")


def test_joint_network_learns_both_tasks_and_reloads(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from erp.models.mlp import MLP

    rng = np.random.default_rng(3)
    n = 1500
    x = pd.DataFrame({"size": rng.normal(0, 1, n), "blocked": rng.integers(0, 2, n).astype(float),
                      "noise": rng.normal(0, 1, n)})
    y = {"effort": np.log1p(np.exp(1.2 + 0.6 * x["size"] + rng.normal(0, 0.2, n))).to_numpy(),
         "risk": (rng.random(n) < np.where(x["blocked"] > 0, 0.8, 0.2)).astype(int)}
    fit, val = np.arange(n) < 1200, np.arange(n) >= 1200
    model = MLP(("effort", "risk")).fit(x[fit], {t: v[fit] for t, v in y.items()}, x[val],
                                        {t: v[val] for t, v in y.items()},
                                        {"lr": 3e-3, "dropout": 0.1, "weight_decay": 1e-4, "batch_size": 128})
    out = model.predict(x[val])
    assert np.corrcoef(out["effort"], y["effort"][val])[0, 1] > 0.8
    blocked = x["blocked"][val].to_numpy() > 0
    assert out["risk"][blocked].mean() > out["risk"][~blocked].mean()
    again = MLP.load(tmp_path, model.save(tmp_path, "joint")).predict(x[val])
    assert all(np.array_equal(out[t], again[t]) for t in out)


def test_holm_and_vargha_delaney():
    assert metrics.holm([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])
    assert metrics.vargha_delaney([1, 2, 3], [1, 2, 3]) == 0.5
    assert metrics.vargha_delaney([4, 5, 6], [1, 2, 3]) == 1.0
    assert metrics.effect_size_label(0.75) == "large" and metrics.effect_size_label(0.52) == "negligible"
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500).astype(bool)
    good, bad = y + rng.normal(0, 0.5, 500), rng.normal(0, 1, 500)
    from sklearn.metrics import roc_auc_score

    result = metrics.paired_bootstrap(roc_auc_score, y, good, bad, runs=200)
    assert result["difference"] > 0.2 and result["low"] > 0 and result["p"] < 0.01


def test_stack_blends_effort_and_combines_risk_logits():
    from erp.arena.predictor import combine

    manifest = {"effort": {"bases": ["a", "b"], "meta": {"weights": [0.25, 0.75], "intercept": 0.1}},
                "risk": {"bases": ["b"], "meta": {"weights": [2.0], "intercept": 0.0}}}
    outputs = {"a": (np.array([1.0]), np.array([0.3])), "b": (np.array([2.0]), np.array([0.5]))}
    log_points, risk = combine(manifest, outputs)
    assert log_points[0] == pytest.approx(0.1 + 0.25 * 1.0 + 0.75 * 2.0)
    assert risk[0] == pytest.approx(0.5)  # logit(0.5) = 0, so the combined logit is the intercept


def test_router_score_and_hard_requirements():
    from erp.arena import report

    row = {"sa": 30.0, "f1": 0.7, "ece": 0.03, "latency_p95": 1.0, "coverage_0.8": 0.81, "coverage_0.9": 0.9}
    assert report.composite(row, report.WEIGHTS) == pytest.approx(
        0.35 * 0.30 + 0.25 * 0.7 + 0.25 * 0.97 + 0.15 * 0.5)
    assert report.failed_requirements(row) == []
    slow = {**row, "latency_p95": 2.5, "ece": 0.12, "coverage_0.8": 0.70}
    assert report.failed_requirements(slow) == ["NFR1 latency", "NFR3 ECE", "NFR3 80% coverage"]
    assert report.winner({"a": 0.6, "b": 0.7}, eligible={"a"}) == "a"
