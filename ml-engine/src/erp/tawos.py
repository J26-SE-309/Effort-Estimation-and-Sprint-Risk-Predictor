"""Reading the exported TAWOS tables and decoding their less obvious fields."""

import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from erp import config


def load(table: str, columns: list[str] | None = None, filters=None, parquet_dir: Path | None = None) -> pd.DataFrame:
    """Load one exported TAWOS table, optionally only some columns / rows (pyarrow filters)."""
    path = (parquet_dir or config.TAWOS_PARQUET_DIR) / f"{table}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run erp-export-tawos first.")
    return pq.read_table(path, columns=columns, filters=filters).to_pandas()


_SPRINT_ID = re.compile(r"\d+")


def unquote(value: str | None) -> str | None:
    """Undo the CSV quoting on TAWOS Issue text fields (Title, Description).

    The dump stores them as '"Fix stream failover "': wrapped in double quotes, inner quotes doubled and
    line breaks flattened to spaces. Change_Log strings are stored as typed, so both need this step (and
    whitespace collapsing) before they can be compared.
    """
    if value is None or pd.isna(value):
        return None
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1].replace('""', '"')
    return value


# Projects name the same priorities differently (Major, Major - P3, Medium), and Atlassian renamed Major to
# Medium, Minor to Low and so on without logging it, so priorities are compared on one five-level scale.
PRIORITY_LEVELS = {
    "Blocker": "highest", "Blocker - P1": "highest", "Highest": "highest",
    "Critical": "high", "Critical - P2": "high", "High": "high",
    "Major": "medium", "Major - P3": "medium", "Medium": "medium",
    "Minor": "low", "Minor - P4": "low", "Low": "low",
    "Trivial": "lowest", "Trivial - P5": "lowest", "Lowest": "lowest",
}


def priority_level(name: str | None) -> str:
    """Map a Jira priority name to highest / high / medium / low / lowest, or 'unknown'."""
    if name is None or pd.isna(name):
        return "unknown"
    return PRIORITY_LEVELS.get(name.strip(), "unknown")


def parse_sprint_ids(value: str | None) -> list[int]:
    """Decode a Change_Log 'Sprint' value into sprint Jira IDs, in the order Jira lists them.

    Jira stores the full list of sprints an issue has belonged to, not just the
    newest one: adding an issue to a second sprint changes '103' to '103, 106'.
    These are Jira sprint IDs (Sprint.JiraID), not TAWOS Sprint.ID values.
    """
    if value is None or pd.isna(value):
        return []
    return [int(match) for match in _SPRINT_ID.findall(value)]
