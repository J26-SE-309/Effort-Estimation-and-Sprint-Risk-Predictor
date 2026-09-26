"""What an issue's fields looked like at a given moment, replayed from the Change_Log."""

import pandas as pd


def value_at(changes: pd.DataFrame, at: pd.Series, current: pd.Series) -> pd.DataFrame:
    """The value of one field for each issue at the time given in `at`, and whether it changed afterwards.

    changes: Change_Log rows for one field (ID, Issue_ID, Creation_Date, From_String).
    at:      the moment per issue (index Issue_ID).
    current: the field's value today per issue (index Issue_ID), e.g. from the Issue table.

    Every change records the value it replaced, so the value at time t is the From_String of the first
    change made after t. With no change after t, the value at t is today's value. A change made at exactly
    t counts as already made.
    """
    rows = changes.merge(at.rename("_at"), left_on="Issue_ID", right_index=True)
    after = rows[rows["Creation_Date"] > rows["_at"]].sort_values(["Issue_ID", "Creation_Date", "ID"])
    replaced = after.groupby("Issue_ID").head(1).set_index("Issue_ID")["From_String"]
    changed_later = pd.Series(at.index.isin(replaced.index), index=at.index)
    value = current.reindex(at.index).astype(object).where(~changed_later, replaced.reindex(at.index))
    return pd.DataFrame({"value": value, "changed_later": changed_later})
