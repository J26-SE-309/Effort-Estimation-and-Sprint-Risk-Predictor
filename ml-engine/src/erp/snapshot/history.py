"""What an issue's fields looked like at a given moment, replayed from the Change_Log."""

import pandas as pd


def value_at(changes: pd.DataFrame, at: pd.Series, current: pd.Series) -> pd.DataFrame:
    """The value of one field for each issue at the time given in `at`, and whether it changed afterwards.

    changes: Change_Log rows for one field (ID, Issue_ID, Creation_Date, From_String, To_String).
    at:      the moment per issue (index Issue_ID).
    current: the field's value today per issue (index Issue_ID), e.g. from the Issue table.

    The value at time t is what the last change made at or before t set it to. With no change by then, it is
    what the first change after t replaced (fields filled in when the issue was created have no log entry),
    and with no change at all it is today's value. Reading forwards first matters when the log misses an
    event: a story resolved in 2016 and silently reopened in 2018 still counts as resolved in 2016.
    Returns value, from_log (False when today's value was used) and changed_later.
    """
    rows = changes.merge(at.rename("_at"), left_on="Issue_ID", right_index=True)
    rows = rows.sort_values(["Issue_ID", "Creation_Date", "ID"])
    before = rows[rows["Creation_Date"] <= rows["_at"]].groupby("Issue_ID").tail(1).set_index("Issue_ID")["To_String"]
    after = rows[rows["Creation_Date"] > rows["_at"]].groupby("Issue_ID").head(1).set_index("Issue_ID")["From_String"]

    has_before = pd.Series(at.index.isin(before.index), index=at.index)
    has_after = pd.Series(at.index.isin(after.index), index=at.index)
    value = current.reindex(at.index).astype(object)
    value = value.where(~has_after, after.reindex(at.index))
    value = value.where(~has_before, before.reindex(at.index))
    return pd.DataFrame({"value": value, "from_log": has_before | has_after, "changed_later": has_after})
