"""Features for stories arriving at the service, computed the way the training pipeline computed them.

A live request knows much less than TAWOS recorded. Every catalogue feature is therefore:
  - taken from the request when the caller knows it (links, team history, sprint context);
  - computed from the story text when it can be, with the same code as training (erp.features.text);
  - otherwise left missing (NaN). LightGBM, XGBoost and CatBoost handle missing values natively; the other
    learners fill them with their training medians (TabularPrep).

The upstream groups (Components 1-3) come from the component when the gateway supplies its signal, and from
the same text proxies the models were trained on when it does not (FR17): the story is still predicted, and
the response says which source was used.

Stories arrive as plain dicts with the API's field names (app/schemas.py of the backend).
"""

import numpy as np
import pandas as pd

from erp import tawos
from erp.features import catalog
from erp.features import text as words
from erp.snapshot.text import clean_text

QUALITY_FIELDS = ("ambiguity_score", "vague_term_count", "missing_info_flag_count")
AC_FIELDS = ("ac_completeness_score", "invest_compliance_flags")
TRACE_FIELDS = ("traceability_coverage_pct", "unlinked_artifact_count", "has_linked_tests")
INVEST = ("independent", "valuable", "testable")

# Where each feature group came from, per story: component (Components 1-3), request (the caller), history (the
# project's stored sprint records, erp.serving.history), text (computed here from the story), proxy (our stand-in
# for a component), missing (unknown, the models see NaN).
SOURCES = ("component", "request", "history", "text", "proxy", "missing")


def full_description(description: str | None, criteria: list[str] | None) -> str:
    """The acceptance criteria join the description under a heading, as they appear in Jira descriptions."""
    text = (description or "").rstrip()
    if criteria:
        text += ("\n\n" if text else "") + "Acceptance criteria:\n" + "\n".join(f"- {c}" for c in criteria)
    return text


def _get(mapping: dict | None, key: str):
    value = (mapping or {}).get(key)
    return np.nan if value is None else value


def build(project_id: str, stories: list[dict], team: dict | None = None,
          sprint: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(features, sources): one row per story with every catalogue feature plus title and description_text, and
    the source of each feature group."""
    index = pd.Index([s["story_id"] for s in stories], name="story_id")
    raw = [full_description(s.get("description"), s.get("acceptance_criteria")) for s in stories]
    base = pd.DataFrame({
        "title": [s["title"] for s in stories],
        "description": raw,
        "description_text": [clean_text(r) for r in raw],
        "issue_type": [s.get("issue_type") or "" for s in stories],
    }, index=index)
    found = words.text_features(base)
    upstream = [s.get("upstream") or {} for s in stories]
    f = pd.DataFrame(index=index)
    sources = pd.DataFrame(index=index)

    # Story text
    for name in ("title_length", "description_length", "description_has_code", "user_story_format", "mentions_tests"):
        f[name] = found[name]
    sources["text"] = "text"

    # Requirement quality (Component 1 or its text proxies)
    for name in QUALITY_FIELDS:
        f[name] = [u.get(name) if u.get(name) is not None else found.at[i, name] for i, u in zip(index, upstream,
                                                                                                    strict=True)]
    sources["requirement_quality"] = ["component" if any(u.get(n) is not None for n in QUALITY_FIELDS) else "proxy"
                                      for u in upstream]

    # Dependencies (from the request; 0 when not given)
    for name in ("blocker_count", "dep_out_degree", "dep_in_degree"):
        f[name] = [int(s.get(name) or 0) for s in stories]
    sources["dependencies"] = "request"

    # Traceability (Component 3, or what the request says about links and the epic)
    f["has_epic"] = [_get(s, "has_epic") for s in stories]
    f["linked_issue_count"] = [s["linked_issue_count"] if s.get("linked_issue_count") is not None
                               else s.get("dep_out_degree", 0) + s.get("dep_in_degree", 0) for s in stories]
    tests = [u["has_linked_tests"] if u.get("has_linked_tests") is not None else np.nan for u in upstream]
    f["has_linked_tests"] = tests
    known = pd.DataFrame({"epic": f["has_epic"], "linked": f["linked_issue_count"] > 0, "tests": f["has_linked_tests"]})
    all_known = known[["epic", "tests"]].notna().all(axis=1)
    traces = known.astype(float).where(all_known)
    f["traceability_coverage_pct"] = [u["traceability_coverage_pct"] if u.get("traceability_coverage_pct") is not None
                                      else value for u, value in zip(upstream, traces.mean(axis=1).round(4),
                                                                     strict=True)]
    f["unlinked_artifact_count"] = [u["unlinked_artifact_count"] if u.get("unlinked_artifact_count") is not None
                                    else value for u, value in zip(upstream, 3 - traces.sum(axis=1, min_count=3),
                                                                   strict=True)]
    sources["traceability"] = ["component" if any(u.get(n) is not None for n in TRACE_FIELDS)
                               else "request" if bool(ok) else "missing" for u, ok in zip(upstream, all_known,
                                                                                         strict=True)]

    # Acceptance criteria and INVEST (Component 2 or text proxies)
    completeness = [u.get("ac_completeness_score") for u in upstream]
    f["ac_completeness_score"] = [c if c is not None else found.at[i, "ac_completeness_score"]
                                  for i, c in zip(index, completeness, strict=True)]
    f["has_acceptance_criteria"] = [(c > 0) if c is not None else bool(found.at[i, "has_acceptance_criteria"])
                                    for i, c in zip(index, completeness, strict=True)]
    proxy_invest = pd.DataFrame({
        "independent": f["blocker_count"] == 0,
        "valuable": found["user_story_format"] | found["states_goal"],
        "testable": f["has_acceptance_criteria"].astype(bool) | f["has_linked_tests"].fillna(False).astype(bool)
        | found["mentions_tests"],
    }, index=index)
    for flag in INVEST:
        f[f"invest_{flag}"] = [_invest(u.get("invest_compliance_flags"), flag, proxy_invest.at[i, flag])
                               for i, u in zip(index, upstream, strict=True)]
    sources["acceptance_criteria"] = ["component" if any(u.get(n) is not None for n in AC_FIELDS) else "proxy"
                                      for u in upstream]

    # Team history (the caller's summary of the team's past sprints, when known)
    f["team_velocity_rolling"] = _get(team, "velocity_mean")
    f["velocity_variance"] = _get(team, "velocity_variance")
    f["history_sprints"] = _get(team, "closed_sprints")
    f["mean_cycle_time_hours"] = _get(team, "mean_cycle_time_hours")
    f["historical_spillover_rate"] = _get(team, "spillover_rate")
    f["reopen_rate"] = _get(team, "reopen_rate")
    known = team and team.get("velocity_mean") is not None
    sources["team_history"] = (team.get("source") or "request") if known else "missing"

    # This sprint: the backlog itself says what else is committed; length, parallel sprints and WIP come from
    # the caller.
    points = pd.Series([s.get("story_points") for s in stories], index=index, dtype=float)
    others = points.fillna(0).sum() - points.fillna(0)
    f["sprint_length_days"] = _get(sprint, "length_days")
    f["days_into_sprint"] = [float(s.get("days_into_sprint") or 0) for s in stories]
    f["parallel_sprints"] = _get(sprint, "parallel_sprints")
    f["sprint_committed_points"] = others if points.notna().any() else np.nan
    velocity = f["team_velocity_rolling"].astype(float)
    f["commitment_to_velocity_ratio"] = f["sprint_committed_points"] / velocity.where(velocity > 0)
    f["wip_at_commitment"] = _get(sprint, "wip")
    f["in_progress_at_commitment"] = [bool(s.get("in_progress")) for s in stories]
    known = sprint and sprint.get("length_days") is not None
    sources["sprint"] = (sprint.get("source") or "request") if known else "missing"

    # Metadata
    f["added_mid_sprint"] = [bool(s.get("added_mid_sprint")) for s in stories]
    f["story_points"] = points
    f["issue_type"] = [s.get("issue_type") or np.nan for s in stories]
    f["priority_level"] = [tawos.priority_level(s.get("priority")) for s in stories]
    f["project_key"] = project_id
    sources["metadata"] = "request"

    features = f[catalog.names()].copy()
    for name in ("has_epic", "has_linked_tests"):
        features[name] = features[name].astype(float)  # True / False / unknown
    features["title"], features["description_text"] = base["title"], base["description_text"]
    return features, sources


def _invest(flags: dict | None, name: str, fallback: bool) -> bool:
    """Component 2's INVEST flag (keys matched loosely, e.g. 'Independent' or 'is_independent'), else our proxy."""
    for key, value in (flags or {}).items():
        if name in key.lower() and value is not None:
            return bool(value)
    return bool(fallback)


def missing_groups(sources: pd.DataFrame) -> pd.Series:
    """How many feature groups a story is missing, for the confidence score. Team history is left out: a team
    without known history is already a cold start there, and counting it twice would punish it twice."""
    return (sources.drop(columns="team_history") == "missing").sum(axis=1)
