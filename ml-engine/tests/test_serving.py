"""The prediction engine behind the service: live features, R1 router, R2 recommendations, A1 simulation."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from erp.features import catalog
from erp.serving import live, recommend, sprint
from erp.serving import router as router_module
from erp.serving.router import MARGIN, Router


def test_live_features_come_from_components_the_request_the_text_or_are_missing():
    stories = [
        {"story_id": "S-1", "title": "As a user I want to export reports so that I can share them",
         "description": "Export as PDF. TBD: page size.", "acceptance_criteria": ["Given a report When I export Then a"
                                                                                  " PDF downloads"],
         "issue_type": "Story", "priority": "Major", "story_points": 3, "blocker_count": 1},
        {"story_id": "S-2", "title": "Fix crash", "description": "", "story_points": None,
         "upstream": {"ambiguity_score": 0.9, "vague_term_count": 4, "missing_info_flag_count": 2,
                      "traceability_coverage_pct": 0.0, "unlinked_artifact_count": 3, "has_linked_tests": False,
                      "invest_compliance_flags": {"Independent": False}}},
    ]
    features, sources = live.build("SYN", stories, team={"velocity_mean": 10.0, "closed_sprints": 4}, sprint=None)
    assert list(features.columns[: len(catalog.names())]) == catalog.names()
    one, two = features.loc["S-1"], features.loc["S-2"]
    assert one["has_acceptance_criteria"] and one["user_story_format"] and one["priority_level"] == "medium"
    assert not one["invest_independent"]  # it has an open blocker
    assert two["ambiguity_score"] == 0.9 and two["unlinked_artifact_count"] == 3 and not two["invest_independent"]
    assert one["sprint_committed_points"] == 0 and two["sprint_committed_points"] == 3  # the others' points
    assert np.isnan(two["story_points"]) and two["team_velocity_rolling"] == 10.0
    assert sources.loc["S-1", "requirement_quality"] == "proxy" and sources.loc["S-2", "requirement_quality"] == \
        "component"
    assert sources.loc["S-1", "traceability"] == "missing" and sources.loc["S-2", "traceability"] == "component"
    assert (sources["sprint"] == "missing").all() and live.missing_groups(sources).tolist() == [2, 1]


@pytest.fixture
def leaderboard(tmp_path):
    for name in ("pooled", "local", "other"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "model.json").write_text("{}", encoding="utf-8")
    board = {"pooled_winner": "pooled",
             "configs": {n: {"eligible": True, "composite": c} for n, c in (("pooled", 0.6), ("local", 0.5),
                                                                           ("other", 0.4))},
             "per_project": {"CLEAR": {"winner": "local", "composite": {"pooled": 0.50, "local": 0.50 + MARGIN}},
                             "CLOSE": {"winner": "local", "composite": {"pooled": 0.50, "local": 0.51}}}}
    (tmp_path / "leaderboard.json").write_text(json.dumps(board), encoding="utf-8")
    return Router(tmp_path)


def test_router_rules(leaderboard):
    assert leaderboard.choose("CLEAR", 10).config == "local"  # the project's own winner, clearly ahead
    assert leaderboard.choose("CLOSE", 10).config == "pooled"  # a lead below the margin is noise
    assert leaderboard.choose("CLEAR", 1).config == "pooled"  # cold start
    assert leaderboard.choose("CLEAR", None).config == "pooled"  # unknown history
    assert leaderboard.choose("NEW", 10).config == "pooled"  # not in the leaderboard
    pinned = leaderboard.choose("CLEAR", 10, pinned="other")
    assert (pinned.config, pinned.mode) == ("other", "pinned")
    with pytest.raises(KeyError):
        leaderboard.choose("CLEAR", 10, pinned="missing")


def test_router_offers_only_configurations_whose_packages_are_installed(tmp_path, monkeypatch):
    configs = {"fasttext-lightgbm": ("fasttext", "lightgbm"), "sbert-lightgbm": ("sbert", "lightgbm"),
               "tfidf-mtl": ("tfidf", "mlp")}
    for name, (encoder, learner) in configs.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "model.json").write_text(json.dumps({"config": {"encoder": encoder, "learner": learner}}),
                                                    encoding="utf-8")
    for name, bases in (("stack", ["fasttext-lightgbm", "sbert-lightgbm"]), ("stack-light", ["fasttext-lightgbm"])):
        (tmp_path / name).mkdir()
        (tmp_path / name / "model.json").write_text(json.dumps({"config": {"encoder": "several", "learner": "stack"},
                                                                "effort": {"bases": bases}}), encoding="utf-8")
    board = {"pooled_winner": "fasttext-lightgbm",
             "configs": {n: {"eligible": True, "composite": 0.5} for n in (*configs, "stack", "stack-light")}}
    (tmp_path / "leaderboard.json").write_text(json.dumps(board), encoding="utf-8")
    find_spec = router_module.importlib.util.find_spec
    monkeypatch.setattr(router_module.importlib.util, "find_spec",
                        lambda name: None if name == "torch" else find_spec(name))
    assert Router(tmp_path).available == ["fasttext-lightgbm", "stack-light"]  # the image built without torch


def test_a_copied_arena_folder_loads_its_own_encoders(tmp_path, monkeypatch):
    """The service image copies the models elsewhere; nothing may be read from the source tree's path."""
    pytest.importorskip("gensim")
    from erp.arena import predictor
    from erp.models.encoders import FastTextEncoder
    from erp.serving.engine import default_models_dir

    source = default_models_dir()
    if not (source / "fasttext-lightgbm").exists():
        pytest.skip("the trained models are not present")
    shutil.copytree(source / "fasttext-lightgbm", tmp_path / "fasttext-lightgbm")
    shutil.copytree(source / "encoders" / "fasttext", tmp_path / "encoders" / "fasttext")
    read, load = [], FastTextEncoder.load
    monkeypatch.setattr(FastTextEncoder, "load", classmethod(lambda cls, directory: read.append(directory) or
                                                              load(directory)))
    predictor.load(tmp_path / "fasttext-lightgbm")
    assert read == [tmp_path / "encoders" / "fasttext"]


def _story(**overrides) -> pd.Series:
    values = {"title": "Improve search", "description_text": "Make it fast and user-friendly.",
              "has_acceptance_criteria": False, "ac_completeness_score": 0.0, "has_linked_tests": np.nan,
              "blocker_count": 0, "dep_out_degree": 0, "commitment_to_velocity_ratio": np.nan}
    return pd.Series({**values, **overrides})


def test_recommendations_only_for_reasons_the_model_gave_and_only_when_actionable():
    risky = {"predicted_story_points": 3, "interval_low": 1, "interval_high": 6, "effort_category": "medium",
             "sprint_risk_level": "high"}
    reasons = [{"factor": "Requirement ambiguity"}, {"factor": "Dependencies and blockers"},
               {"factor": "Sprint load and timing"}]
    actions = [r["action"] for r in recommend.for_story(risky, _story(), reasons)]
    assert actions == ["refine_requirement"]  # no dependencies to resolve, sprint load unknown
    loaded = _story(blocker_count=2, commitment_to_velocity_ratio=1.4)
    actions = [r["action"] for r in recommend.for_story(risky, loaded, reasons)]
    assert actions == ["refine_requirement", "resolve_dependency", "reduce_sprint_scope"]
    calm = {**risky, "sprint_risk_level": "low"}
    assert recommend.for_story(calm, loaded, reasons) == []
    huge = {**calm, "predicted_story_points": 20, "effort_category": "extra_large"}
    assert [r["action"] for r in recommend.for_story(huge, loaded, reasons)] == ["split_story"]
    assert [recommend.effort_category(p) for p in (1, 2, 3, 5, 8, 13)] == [
        "small", "small", "medium", "medium", "large", "extra_large"]


def test_sprint_simulation():
    log_points = np.log1p([3, 5, 8, 2])
    high = np.expm1(log_points + 0.5)
    probability = [0.2, 0.5, 0.8, 0.1]
    unknown = sprint.simulate(log_points, high, probability, capacity_mean=None)
    assert unknown["overcommit_probability"] is None and unknown["expected_stories_at_risk"] == 1.6
    tight = sprint.simulate(log_points, high, probability, capacity_mean=10, capacity_variance=4)
    roomy = sprint.simulate(log_points, high, probability, capacity_mean=40, capacity_variance=4)
    assert tight["overcommit_probability"] > 0.7 and tight["sprint_risk_level"] == "high"
    assert roomy["overcommit_probability"] < 0.05 and roomy["sprint_risk_level"] == "low"
    assert tight["committed_points"]["p10"] < 18 < tight["committed_points"]["p90"]
    assert sprint.simulate(log_points, high, probability, 10, 4) == tight  # fixed seed: reproducible
    advice = recommend.sprint_level(tight)
    assert advice and advice[0]["action"] == "reduce_sprint_scope"
