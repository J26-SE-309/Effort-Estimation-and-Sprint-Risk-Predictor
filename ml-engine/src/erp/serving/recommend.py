"""R2: planning recommendations (FR15; ML guide 4.9 and Appendix A).

The rule that keeps recommendations honest: an action is suggested only when its planning factor is among the
reasons driving this story's prediction, so every recommendation points back to the explanation next to it.
  split_story              the effort estimate is in the extra-large band; or "Story size and content" is a reason
                           and the estimate, or the team's own, is in the large band
  refine_requirement       "Requirement ambiguity" is a reason for the risk; or "Story size and content" is, and
                           the story says almost nothing (under SHORT_DESCRIPTION words)
  add_acceptance_criteria  "Acceptance-criteria gaps" is a reason for the risk and the criteria are not complete
  create_tests             "Test and traceability gaps" is a reason for the risk
  resolve_dependency       "Dependencies and blockers" is a reason, or "Acceptance-criteria gaps" is (its INVEST
                           'independent' flag is about open blockers), and the story really has dependencies
  reduce_sprint_scope      "Sprint load and timing" or "Team delivery history" is a reason, and the sprint is
                           committed well above the team's recent velocity; or "Sprint load and timing" is, and the
                           story was added after the sprint started
  plan_buffer              "Team delivery history" is a reason and the sprint is not overloaded: the team spills
                           over often, or the service knows too little of its history yet
Risk-driven actions are only suggested for stories at medium or high risk: a low-risk story needs no fixing.
"Story metadata" (issue type, priority) has no action: it describes the story rather than something to change.
"""

import numpy as np
import pandas as pd

from erp.features import text as words

EFFORT_BANDS = {"small": 2.5, "medium": 6.5, "large": 10.5}  # between Fibonacci steps: <=2, 3-5, 8, >=13
OVERLOAD_RATIO = 1.1  # committed points / recent velocity above this is 'well above 1' (ML guide 4.9)
SHORT_DESCRIPTION = 15  # words: a story this short leaves the team guessing what it means
COMPLETE_CRITERIA = 0.8  # acceptance criteria at least this complete leave nothing to add
HIGH_SPILLOVER = 0.2  # a team spilling over more than 1 story in 5 recently
COLD_START_SPRINTS = 3  # as the confidence score's cold start (erp.models.confidence)
OWN_FACTOR = {"resolve_dependency": "Dependencies and blockers", "refine_requirement": "Requirement ambiguity",
              "reduce_sprint_scope": "Sprint load and timing"}
ACTION_OF = {  # per factor, the actions that may apply; every one that applies to the story is suggested
    "Requirement ambiguity": ["refine_requirement"],
    "Acceptance-criteria gaps": ["add_acceptance_criteria", "resolve_dependency"],
    "Test and traceability gaps": ["create_tests"],
    "Dependencies and blockers": ["resolve_dependency"],
    "Story size and content": ["split_story", "refine_requirement"],
    "Sprint load and timing": ["reduce_sprint_scope"],
    # an overloaded sprint is the team's history at work; otherwise (only) the team's record calls for some slack
    "Team delivery history": ["reduce_sprint_scope", "plan_buffer"],
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
    factors = {reason["factor"] for reason in risk_reasons}
    for reason in risk_reasons:
        factor = reason["factor"]
        for action in ACTION_OF.get(factor, []):
            taken = {r["action"] for r in out}
            if action in taken or (action == "plan_buffer" and "reduce_sprint_scope" in taken):
                continue
            if action != ACTION_OF[factor][0] and OWN_FACTOR.get(action) in factors - {factor}:
                continue  # it comes with the factor it belongs to
            message = _message(action, story, factor, prediction)
            if message:
                out.append({"action": action, "triggered_by": factor, "message": message})
    return out


def _message(action: str, story: pd.Series, factor: str, prediction: dict) -> str | None:
    """The recommendation's text, or None when the story gives nothing to act on."""
    if action == "split_story":
        own = _number(story.get("story_points"))
        if prediction["effort_category"] == "large":  # extra-large is suggested above, whatever the reasons
            size = f"Estimated at about {prediction['predicted_story_points']:.0f} points"
        elif own is not None and effort_category(own) in ("large", "extra_large"):
            size = f"The team estimates it at {own:.0f} points"
        else:
            return None
        return (f"{size}, a large story for one sprint: the bigger a story, the more often it spills over. Split it "
                "if it holds separate parts.")
    if action == "refine_requirement" and factor == "Story size and content":
        length = _number(story.get("description_length"))
        if length is None or length >= SHORT_DESCRIPTION:
            return None
        said = "has no description" if not length else f"is described in only {length:.0f} words"
        return f"The story {said}. Describe what it should do and for whom, so the team estimates the same work."
    if action == "refine_requirement":
        terms = list(dict.fromkeys(words.vague_terms(f"{story['title']}. {story['description_text']}")))
        detail = f" (vague terms: {', '.join(repr(t) for t in terms[:4])})" if terms else ""
        return f"The requirement's wording is unclear{detail}. Clarify what 'done' means before committing to it."
    if action == "add_acceptance_criteria":
        if not story["has_acceptance_criteria"]:
            return "The story has no acceptance criteria. Add testable criteria (e.g. Given / When / Then)."
        completeness = _number(story.get("ac_completeness_score"))
        if completeness is not None and completeness >= COMPLETE_CRITERIA:
            return None  # nothing missing to add: the factor's push comes from elsewhere in the group (INVEST)
        shown = "" if completeness is None else f" (completeness {completeness:.0%})"
        return f"The acceptance criteria look incomplete{shown}. Add the missing cases."
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
        if load is not None and load > OVERLOAD_RATIO:
            return (f"The sprint is committed at {load:.0%} of the team's recent velocity. Move lower-priority "
                    "stories out of the sprint.")
        if factor == "Sprint load and timing" and story.get("added_mid_sprint"):
            return ("It was added after the sprint started. Agree what leaves the sprint to make room for it, or "
                    "plan it for the next sprint.")
        return None
    if action == "plan_buffer":
        sprints = _number(story.get("history_sprints")) or 0
        spillover = _number(story.get("historical_spillover_rate"))
        if sprints < COLD_START_SPRINTS:
            known = "no closed sprints" if not sprints else f"only {sprints:.0f} closed sprint{'s' * (sprints > 1)}"
            return (f"The team has {known} on record, so the risk leans on how other teams delivered. Keep some "
                    "capacity free for this story and check its estimate together at planning.")
        if spillover is not None and spillover > HIGH_SPILLOVER:
            return (f"The team spilled over {spillover:.0%} of its recent stories. Keep some capacity free for this "
                    "story, or pair on it so it does not slip.")
        return None
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
