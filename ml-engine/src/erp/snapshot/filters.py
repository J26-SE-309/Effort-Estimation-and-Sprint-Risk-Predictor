"""Which TAWOS issues count as stories for this component (ML guide section 6.4), and the filtering log."""

import pandas as pd

# Agreed 2026-09-26 ("option B"): the guide's Story, Task and Bug, plus Improvement and New Feature,
# which several projects use for the same kind of sprint work. Epics, sub-tasks and the small
# mixed types (Suggestion, Build Failure, Question, ...) are left out.
STORY_TYPES = ("Story", "Task", "Bug", "Improvement", "New Feature")
SP_RANGE = (1, 100)
MIN_PROJECT_STORIES = 100  # a project needs this many usable stories for its own model

# "Projects that really used sprints" (guide 6.4): enough dated sprints, and a real share of the project's
# estimated issues went through them. The projects below these lines tried sprints only briefly (1-8% of
# their estimated issues were committed to a dated sprint; the next lowest project is at 34%).
PROJECT_MIN_SPRINTS = 10
PROJECT_MIN_SPRINT_SHARE = 0.20
NOT_STORIES = ("Epic", "Sub-task", "Technical task")


def projects_using_sprints(issues: pd.DataFrame, commitments: pd.DataFrame) -> pd.DataFrame:
    """Per project: how many dated sprints it committed issues to, and what share of its estimated issues.

    issues is indexed by issue ID (Project_ID, Type, Story_Point); commitments is first_commitments indexed by
    issue ID (commitment_status, commitment_project, Sprint_ID). Computed on the whole project, not on the
    stories left after the other filters, so the answer does not depend on the order of the filtering steps.
    """
    ok = commitments[commitments["commitment_status"] == "ok"]
    estimated = issues[issues["Story_Point"].notna() & ~issues["Type"].isin(NOT_STORIES)]
    committed = pd.Series(estimated.index.isin(ok.index), index=estimated.index)
    out = pd.DataFrame({
        "sprints": ok.groupby("commitment_project")["Sprint_ID"].nunique(),
        "share": committed.groupby(estimated["Project_ID"]).mean(),
    }).fillna(0)
    out["uses_sprints"] = (out["sprints"] >= PROJECT_MIN_SPRINTS) & (out["share"] >= PROJECT_MIN_SPRINT_SHARE)
    return out


def story_candidates(issues: pd.DataFrame) -> pd.Series:
    """Rows that can become training stories: a story type, plausible story points and a title.

    Uses today's Type and Story_Point. The snapshot applies the same rules to the values the issue had
    when it was committed.
    """
    return (
        issues["Type"].isin(STORY_TYPES)
        & issues["Story_Point"].between(*SP_RANGE)
        & issues["Title"].fillna("").str.strip().ne("")
    )


class FilterLog:
    """Applies filters one after another and records how many rows each one removed.

    The finished table goes straight into the methodology chapter ("started with N stories, removed
    epics and sub-tasks (-A), ..."), so every step has a plain-English label.
    """

    def __init__(self, frame: pd.DataFrame, label: str):
        self.frame = frame
        self.steps = [{"Step": label, "Removed": "", "Remaining": len(frame)}]

    def keep(self, mask: pd.Series, label: str) -> pd.DataFrame:
        before = len(self.frame)
        self.frame = self.frame[mask.reindex(self.frame.index, fill_value=False).astype(bool)]
        self.steps.append({"Step": label, "Removed": before - len(self.frame), "Remaining": len(self.frame)})
        return self.frame

    def table(self) -> pd.DataFrame:
        table = pd.DataFrame(self.steps)
        table["Removed"] = [f"−{v:,}" if v != "" else "" for v in table["Removed"]]
        table["Remaining"] = table["Remaining"].map("{:,}".format)
        return table
