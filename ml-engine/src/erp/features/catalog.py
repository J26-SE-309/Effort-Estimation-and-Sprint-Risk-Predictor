"""Every feature the models may use: its group, what it means, and why it was known at commitment.

The groups follow ML guide section 5 and proposal Appendix A. The three upstream groups carry the names of
the backend's UpstreamSignals contract, so the proxies computed here can be swapped for Components 1-3's
batch scores without renaming anything; the H2 ablation drops a whole group at a time.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    name: str
    group: str
    meaning: str
    known_because: str


FEATURES = [
    # Story text (the text itself goes to the encoders E1-E4; these are simple numbers about it)
    Feature("title_length", "text", "Words in the title", "Title as it was at commitment"),
    Feature("description_length", "text", "Words in the description (code removed)", "Description at commitment"),
    Feature("description_has_code", "text", "The description has pasted code or logs", "Description at commitment"),
    Feature("user_story_format", "text", "Written as 'As a ... I want ...'", "Text at commitment"),
    Feature("mentions_tests", "text", "The description mentions unit / integration / regression tests or a "
            "test plan", "Text at commitment"),
    # Component 1 proxies (Requirement Quality)
    Feature("ambiguity_score", "requirement_quality", "Share of sentences with a vague term (0-1)",
            "Text at commitment; proxy for Component 1"),
    Feature("vague_term_count", "requirement_quality", "Number of vague terms", "Text at commitment; proxy"),
    Feature("missing_info_flag_count", "requirement_quality", "Signs of missing information (0-4): short "
            "description, placeholders, bug without steps, story without user or goal", "Text at commitment; proxy"),
    # Component 2 proxies (Story Refinement)
    Feature("has_acceptance_criteria", "acceptance_criteria", "Acceptance criteria present", "Text at commitment"),
    Feature("ac_completeness_score", "acceptance_criteria", "Number of criteria / 3, capped at 1 (0 without "
            "criteria)", "Text at commitment; proxy for Component 2"),
    Feature("invest_independent", "acceptance_criteria", "INVEST 'independent': no open blocker",
            "Links and blocker status at commitment"),
    Feature("invest_valuable", "acceptance_criteria", "INVEST 'valuable': states a user or a goal ('so that')",
            "Text at commitment"),
    Feature("invest_testable", "acceptance_criteria", "INVEST 'testable': acceptance criteria, linked tests or "
            "tests mentioned", "Text and links at commitment"),
    # Component 3 proxies (Traceability)
    Feature("traceability_coverage_pct", "traceability", "Share of three traces present: epic, any linked issue, "
            "linked tests", "Links and Epic Link at commitment; proxy for Component 3"),
    Feature("unlinked_artifact_count", "traceability", "How many of the three traces are missing",
            "Links and Epic Link at commitment; proxy"),
    Feature("has_linked_tests", "traceability", "Linked to a Test issue or an issue about tests",
            "Links at commitment"),
    Feature("has_epic", "traceability", "Belongs to an epic", "Epic Link at commitment"),
    Feature("linked_issue_count", "traceability", "Issues linked to it (any link type)", "Links at commitment"),
    # Dependencies
    Feature("blocker_count", "dependencies", "Open issues it depends on", "Links and the blockers' resolution at "
            "commitment"),
    Feature("dep_out_degree", "dependencies", "Issues it depends on ('is blocked by', 'depends on', 'has to be "
            "done after')", "Links at commitment"),
    Feature("dep_in_degree", "dependencies", "Issues that depend on it ('blocks', 'is depended on by', 'has to be "
            "done before')", "Links at commitment"),
    # Team history
    Feature("team_velocity_rolling", "team_history", "Mean points finished in the project's last 3 closed sprints",
            "Only sprints closed before the snapshot time"),
    Feature("velocity_variance", "team_history", "Variance of points finished over the last 5 closed sprints",
            "Only sprints closed before the snapshot time"),
    Feature("history_sprints", "team_history", "How many of the project's sprints had closed before (cold start "
            "when small)", "Only sprints closed before the snapshot time"),
    Feature("mean_cycle_time_hours", "team_history", "Mean hours in progress of the project's last 50 resolved "
            "stories", "Only stories resolved before the snapshot time"),
    Feature("historical_spillover_rate", "team_history", "Share of R1 among the project's last 50 stories whose "
            "sprint had closed", "Only outcomes known before the snapshot time"),
    Feature("reopen_rate", "team_history", "Share of R6 among the project's last 50 stories whose outcome was "
            "known", "Only outcomes known before the snapshot time (close + one sprint)"),
    # This sprint
    Feature("sprint_length_days", "sprint", "Planned sprint length", "Set when the sprint started"),
    Feature("days_into_sprint", "sprint", "Days after the start when the story was added (0 if planned)",
            "The moment of commitment"),
    Feature("parallel_sprints", "sprint", "Other sprints of the project running at the time", "Sprint dates"),
    Feature("sprint_committed_points", "sprint", "Points the other stories of the sprint were committed with at "
            "that moment (this story's own points left out, so M1 cannot read its answer here)",
            "Members and their points at the snapshot time"),
    Feature("commitment_to_velocity_ratio", "sprint", "sprint_committed_points / team_velocity_rolling",
            "Both known at the snapshot time"),
    Feature("wip_at_commitment", "sprint", "Other issues of the sprint already in progress",
            "Their status and resolution at the snapshot time"),
    Feature("in_progress_at_commitment", "sprint", "The story itself was already in progress",
            "Its status at the snapshot time"),
    # Metadata
    Feature("added_mid_sprint", "metadata", "Added after the sprint started", "The moment of commitment"),
    Feature("story_points", "metadata", "Story points at commitment (M1's answer; an input for M2 and M3's risk "
            "head only)", "Replayed at commitment"),
    Feature("issue_type", "metadata", "Story, Task, Bug, Improvement or New Feature", "Type at commitment"),
    Feature("priority_level", "metadata", "highest / high / medium / low / lowest / unknown", "Priority at "
            "commitment"),
    Feature("project_key", "metadata", "Project", "Fixed"),
]

GROUPS = list(dict.fromkeys(f.group for f in FEATURES))
UPSTREAM_GROUPS = ("requirement_quality", "acceptance_criteria", "traceability")


def names(group: str | None = None) -> list[str]:
    return [f.name for f in FEATURES if group is None or f.group == group]
