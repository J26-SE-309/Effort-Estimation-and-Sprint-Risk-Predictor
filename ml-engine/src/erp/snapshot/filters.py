"""Which TAWOS issues count as stories for this component (ML guide section 6.4)."""

import pandas as pd

# Agreed 2026-09-26 ("option B"): the guide's Story, Task and Bug, plus Improvement and New Feature,
# which several projects use for the same kind of sprint work. Epics, sub-tasks and the small
# mixed types (Suggestion, Build Failure, Question, ...) are left out.
STORY_TYPES = ("Story", "Task", "Bug", "Improvement", "New Feature")
SP_RANGE = (1, 100)
MIN_PROJECT_STORIES = 100  # a project needs this many usable stories for its own model


def story_candidates(issues: pd.DataFrame) -> pd.Series:
    """Rows that can become training stories: a story type, plausible story points and a title.

    Uses Issue.Story_Point, which is the final value. The snapshot step re-checks the range
    against the story points the issue had when it was committed.
    """
    return (
        issues["Type"].isin(STORY_TYPES)
        & issues["Story_Point"].between(*SP_RANGE)
        & issues["Title"].fillna("").str.strip().ne("")
    )
