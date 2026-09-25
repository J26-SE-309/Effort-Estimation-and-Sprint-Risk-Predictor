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


def parse_sprint_ids(value: str | None) -> list[int]:
    """Decode a Change_Log 'Sprint' value into sprint Jira IDs, in the order Jira lists them.

    Jira stores the full list of sprints an issue has belonged to, not just the
    newest one: adding an issue to a second sprint changes '103' to '103, 106'.
    These are Jira sprint IDs (Sprint.JiraID), not TAWOS Sprint.ID values.
    """
    if value is None or pd.isna(value):
        return []
    return [int(match) for match in _SPRINT_ID.findall(value)]
