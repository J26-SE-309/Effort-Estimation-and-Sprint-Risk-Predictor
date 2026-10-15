"""What each recommendation is expected to do to the risk: the story's features changed as the action would change
them, predicted again by the same configuration (FR15: recommendations a product owner can weigh).

The action's change, per feature (the story text itself is never rewritten, so its encoding stays as it is):
  add_acceptance_criteria  criteria present and complete; INVEST 'testable'
  refine_requirement       no vague terms, no signs of missing information; a description of at least 40 words
  create_tests             tests linked: one more of the three traces present; INVEST 'testable'
  resolve_dependency       no open blockers or dependencies left; INVEST 'independent'
  reduce_sprint_scope      the sprint committed at the team's velocity (or, for a story added late, planned for
                           the next sprint instead)
  plan_buffer              the sprint committed at BUFFER of the team's velocity
  split_story              the story at half its points (one of the parts)
An action whose change cannot be expressed in the features (e.g. no velocity to reduce the load against) gets no
expected effect. The combination of every suggested action is predicted as well. These are the models' answers
to "what if", not promises: the factors are related, and the story's text is unchanged.
"""

import numpy as np
import pandas as pd

BUFFER = 0.85  # keep 15% of the velocity free
CLEAR_DESCRIPTION = 40  # words


def _number(value) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def _load_to(story: pd.Series, share: float) -> dict | None:
    """Sprint features with the whole sprint committed at share x velocity; None if that is not less than now."""
    velocity, others = _number(story.get("team_velocity_rolling")), _number(story.get("sprint_committed_points"))
    if not velocity or velocity <= 0 or others is None:
        return None
    own = _number(story.get("story_points")) or 0.0
    target = max(0.0, share * velocity - own)
    if target >= others:
        return None
    return {"sprint_committed_points": target, "commitment_to_velocity_ratio": target / velocity}


def change(action: str, story: pd.Series, prediction: dict, triggered_by: str) -> dict | None:
    """The features the action changes, with their new values; None when it cannot be expressed."""
    if action == "add_acceptance_criteria":
        return {"has_acceptance_criteria": True, "ac_completeness_score": 1.0, "invest_testable": True}
    if action == "refine_requirement":
        length = _number(story.get("description_length")) or 0.0
        return {"ambiguity_score": 0.0, "vague_term_count": 0, "missing_info_flag_count": 0,
                "description_length": max(length, CLEAR_DESCRIPTION)}
    if action == "create_tests":
        coverage = _number(story.get("traceability_coverage_pct"))
        unlinked = _number(story.get("unlinked_artifact_count"))
        return {"has_linked_tests": 1.0, "invest_testable": True,
                "traceability_coverage_pct": min(1.0, (coverage or 0.0) + 1 / 3) if coverage is not None else np.nan,
                "unlinked_artifact_count": max(0.0, unlinked - 1) if unlinked is not None else np.nan}
    if action == "resolve_dependency":
        return {"blocker_count": 0, "dep_out_degree": 0, "invest_independent": True}
    if action == "reduce_sprint_scope":
        if triggered_by == "Sprint load and timing" and story.get("added_mid_sprint"):
            return {"added_mid_sprint": False, "days_into_sprint": 0.0}
        return _load_to(story, 1.0)
    if action == "plan_buffer":
        return _load_to(story, BUFFER)
    if action == "split_story":
        return {"story_points": max(1.0, round(prediction["predicted_story_points"] / 2))}
    return None


def scenarios(features: pd.DataFrame, recommendations: list[list[dict]], predictions: list[dict]
              ) -> tuple[pd.DataFrame, list[int], list[tuple[int, str | None]]]:
    """(changed feature rows, the original row of each, and its key: (story position, action) or (position, None)
    for all of the story's actions at once)."""
    rows, origins, keys = [], [], []
    for i, recs in enumerate(recommendations):
        story = features.iloc[i]
        combined: dict = {}
        actions = 0
        for rec in recs:
            changed = change(rec["action"], story, predictions[i], rec["triggered_by"])
            if not changed:
                continue
            rows.append(_with(story, changed))
            origins.append(i)
            keys.append((i, rec["action"]))
            combined.update(changed)
            actions += 1
        if actions > 1:
            rows.append(_with(story, combined))
            origins.append(i)
            keys.append((i, None))
    if not rows:
        return features.iloc[0:0], [], []
    frame = pd.DataFrame(rows)
    frame.index = pd.Index([f"{features.index[o]}#whatif{n}" for n, o in enumerate(origins)], name=features.index.name)
    for column, dtype in features.dtypes.items():  # each row went through a Series of mixed values
        try:
            frame[column] = frame[column].astype(dtype)
        except (TypeError, ValueError):
            pass
    return frame, origins, keys


def _with(story: pd.Series, changed: dict) -> pd.Series:
    row = story.copy()
    for name, value in changed.items():
        row[name] = value
    return row


def take(text, rows: list[int], index: pd.Index):
    """The text encoding of the given story positions, for the changed rows (the text is not changed)."""
    if text is None:
        return None
    if isinstance(text, dict):
        return {name: take(part, rows, index) for name, part in text.items()}
    if isinstance(text, pd.Series):
        return pd.Series(text.iloc[rows].to_numpy(), index=index)
    return text[rows]
