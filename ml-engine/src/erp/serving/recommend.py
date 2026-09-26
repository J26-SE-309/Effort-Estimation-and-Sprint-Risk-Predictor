"""R2: planning recommendations (FR15; ML guide 4.9 and Appendix A).

The rule that keeps recommendations honest: an action is suggested only when its planning factor is among the
reasons driving this story's prediction, so every recommendation points back to the explanation next to it.
  split_story              the effort estimate is in the extra-large band
  refine_requirement       "Requirement ambiguity" is a reason for the risk
  add_acceptance_criteria  "Acceptance-criteria gaps" is a reason for the risk
  create_tests             "Test and traceability gaps" is a reason for the risk
  resolve_dependency       "Dependencies and blockers" is a reason, and the story really has dependencies
  reduce_sprint_scope      "Sprint load and timing" or "Team delivery history" is a reason, and the sprint is
                           committed well above the team's recent velocity
Risk-driven actions are only suggested for stories at medium or high risk: a low-risk story needs no fixing.
"""

import numpy as np
import pandas as pd

from erp.features import text as words

EFFORT_BANDS = {"small": 2.5, "medium": 6.5, "large": 10.5}  # between Fibonacci steps: <=2, 3-5, 8, >=13
OVERLOAD_RATIO = 1.1  # committed points / recent velocity above this is 'well above 1' (ML guide 4.9)
ACTION_OF = {
    "Requirement ambiguity": "refine_requirement",
    "Acceptance-criteria gaps": "add_acceptance_criteria",
    "Test and traceability gaps": "create_tests",
    "Dependencies and blockers": "resolve_dependency",
    "Sprint load and timing": "reduce_sprint_scope",
    "Team delivery history": "reduce_sprint_scope",  # context only; it supports reducing the scope
}


def effort_category(points: float) -> str:
    """Small / medium / large / extra-large, derived from the estimate so the two never disagree (ML guide 4.1)."""
    for band, limit in EFFORT_BANDS.items():
        if points < limit:
            return band
    return "extra_large"


def _number(value) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def for_story(prediction: dict, story: pd.Series, risk_reasons: list[dict]) -> list[dict]:
    """prediction: predicted_story_points, interval low / high, effort_category, sprint_risk_level.
    story: the story's features, title and description_text (erp.serving.live)."""
    out = []
    if prediction["effort_category"] == "extra_large":
        out.append({"action": "split_story", "triggered_by": "Story size and content", "message": (
            f"Estimated at about {prediction['predicted_story_points']:.0f} points (likely "
            f"{prediction['interval_low']:.0f}-{prediction['interval_high']:.0f}). Split it into smaller stories "
            "that can each be finished within one sprint.")})
    if prediction["sprint_risk_level"] == "low":
        return out
    for reason in risk_reasons:
        factor = reason["factor"]
        action = ACTION_OF.get(factor)
        message = _message(action, story) if action else None
        if message and all(r["action"] != action for r in out):
            out.append({"action": action, "triggered_by": factor, "message": message})
    return out


def _message(action: str, story: pd.Series) -> str | None:
    """The recommendation's text, or None when the story gives nothing to act on."""
    if action == "refine_requirement":
        terms = list(dict.fromkeys(words.vague_terms(f"{story['title']}. {story['description_text']}")))
        detail = f" (vague terms: {', '.join(repr(t) for t in terms[:4])})" if terms else ""
        return f"The requirement's wording is unclear{detail}. Clarify what 'done' means before committing to it."
    if action == "add_acceptance_criteria":
        if not story["has_acceptance_criteria"]:
            return "The story has no acceptance criteria. Add testable criteria (e.g. Given / When / Then)."
        return (f"The acceptance criteria look incomplete (completeness {story['ac_completeness_score']:.0%}). "
                "Add the missing cases.")
    if action == "create_tests":
        if story["has_linked_tests"] == 1:
            return None
        return "No tests are linked to this story. Create or link the tests that will show it works."
    if action == "resolve_dependency":
        blockers, depends = int(story["blocker_count"]), int(story["dep_out_degree"])
        if blockers:
            them = "them" if blockers > 1 else "it"
            plural = "s" * (blockers > 1)
            return f"It waits on {blockers} open blocker{plural}. Resolve {them} first or plan around {them}."
        if depends:
            return f"It depends on {depends} other issue{'s' * (depends > 1)}. Make sure they are finished first."
        return None
    if action == "reduce_sprint_scope":
        load = sprint_load(story)
        if load is None or load <= OVERLOAD_RATIO:
            return None
        return (f"The sprint is committed at {load:.0%} of the team's recent velocity. Move lower-priority stories "
                "out of the sprint.")
    return None


def sprint_load(story: pd.Series) -> float | None:
    """The whole sprint's committed points over the team's recent velocity. The model's own feature
    (commitment_to_velocity_ratio) leaves the story's points out, so the effort model cannot read its answer
    there; a sentence about the sprint counts every story, so all of a sprint's stories show the same number."""
    velocity, others = _number(story.get("team_velocity_rolling")), _number(story.get("sprint_committed_points"))
    if not velocity or velocity <= 0 or others is None:
        return None
    return (others + (_number(story.get("story_points")) or 0.0)) / velocity


def sprint_level(simulation: dict) -> list[dict]:
    """Sprint-wide advice from the A1 simulation (FR16)."""
    probability = simulation.get("overcommit_probability")
    if probability is None or probability < 0.5:
        return []
    return [{"action": "reduce_sprint_scope", "triggered_by": "Sprint load and timing", "message": (
        f"In {probability:.0%} of simulated sprints the committed work exceeds the team's capacity (by about "
        f"{simulation['expected_overflow_points']:.0f} points on average). Move about "
        f"{np.ceil(simulation['overflow_if_over_p50']):.0f} points out of the sprint.")}]
