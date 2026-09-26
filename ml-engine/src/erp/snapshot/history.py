"""What an issue's fields and links looked like at a given moment, replayed from the Change_Log."""

import re

import pandas as pd

# How the Change_Log writes a link: 'This issue is blocked by ABC-12' (To_String when added, From_String
# when removed). The phrase is the link type as seen from this issue.
LINK = re.compile(r"^This issue (.+?) ([A-Z][A-Z0-9_]*-\d+)$")


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


def values_at(changes: pd.DataFrame, pairs: pd.DataFrame, current: pd.Series) -> pd.Series:
    """value_at for many (issue, moment) pairs at once, e.g. every issue of a sprint at several moments.

    pairs has columns Issue_ID and at (any number of rows per issue); the result is aligned with its index.
    Same rules as value_at: the last change at or before the moment, else what the first later change
    replaced, else today's value.
    """
    ns = "datetime64[ns]"
    log = changes.assign(at=changes["Creation_Date"].astype(ns)).sort_values(["at", "ID"])
    wanted = pairs[["Issue_ID", "at"]].assign(at=pairs["at"].astype(ns), _row=range(len(pairs)))
    wanted = wanted[wanted["at"].notna()].sort_values("at")
    before = pd.merge_asof(wanted, log[["Issue_ID", "at", "To_String"]], on="at", by="Issue_ID",
                           direction="backward").set_index("_row")
    after = pd.merge_asof(wanted, log[["Issue_ID", "at", "From_String"]].assign(_found=True), on="at",
                          by="Issue_ID", direction="forward", allow_exact_matches=False).set_index("_row")
    has_before = pd.merge_asof(wanted, log[["Issue_ID", "at"]].assign(_found=True), on="at", by="Issue_ID",
                               direction="backward").set_index("_row")["_found"].eq(True)

    rows = pd.RangeIndex(len(pairs))
    value = pd.Series(current.reindex(pairs["Issue_ID"]).to_numpy(), index=rows, dtype=object)
    value = value.where(~after["_found"].eq(True).reindex(rows, fill_value=False), after["From_String"].reindex(rows))
    value = value.where(~has_before.reindex(rows, fill_value=False), before["To_String"].reindex(rows))
    value[pairs["at"].isna().to_numpy()] = None
    return pd.Series(value.to_numpy(), index=pairs.index)


def link_periods(link_changes: pd.DataFrame, issue_ids: pd.Series) -> pd.DataFrame:
    """When each link of each issue existed: one row per link and period (removed_at NaT while it exists).

    link_changes holds Change_Log rows for Field == 'Link' (ID, Issue_ID, From_String, To_String,
    Creation_Date). issue_ids maps issue keys to TAWOS issue IDs; targets outside TAWOS get NA. Links made
    when the issue was created have no log entry and are missed (0.5% of blocking links).
    """
    rows = []
    log = link_changes.sort_values(["Issue_ID", "Creation_Date", "ID"])
    for issue_id, changes in log.groupby("Issue_ID", sort=False):
        open_links: dict[tuple[str, str], pd.Timestamp] = {}
        for when, old, new in zip(changes["Creation_Date"], changes["From_String"], changes["To_String"],
                                  strict=True):
            removed = LINK.match(old) if isinstance(old, str) else None
            added = LINK.match(new) if isinstance(new, str) else None
            if removed:
                link = (removed.group(1).lower(), removed.group(2))
                if link in open_links:
                    rows.append((issue_id, *link, open_links.pop(link), when))
            if added:
                link = (added.group(1).lower(), added.group(2))
                open_links.setdefault(link, when)
        rows += [(issue_id, *link, since, pd.NaT) for link, since in open_links.items()]
    links = pd.DataFrame(rows, columns=["Issue_ID", "phrase", "target_key", "added_at", "removed_at"])
    links["removed_at"] = pd.to_datetime(links["removed_at"])
    links["Target_ID"] = links["target_key"].map(issue_ids).astype("Int64")
    return links
