"""Export every TAWOS table from the restored MySQL database to Parquet.

The Parquet files are an exact copy of the database: nothing is filtered,
cleaned or renamed here, so every later step can be traced back to the source.
Column types come from the MySQL table definitions instead of being guessed
from the data, so a chunk that happens to be all NULL cannot change a type.

A _manifest.json next to the files records row counts, the MySQL version and
the SHA-256 of the dump, so a result can be tied to the exact data it used.

Usage (with the tawos-mysql container running and the dump imported):
    erp-export-tawos                      # all tables
    erp-export-tawos --tables Issue Sprint
"""

import argparse
import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pymysql
import pymysql.cursors

from erp import config

TAWOS_TABLES = [
    "Repository", "Project", "Sprint", "Version", "Component", "User",
    "Issue", "Issue_Link", "Issue_Component", "Fix_Version", "Affected_Version",
    "Change_Log", "Comment",
]

_ARROW_TYPES = {
    "int": pa.int64(),
    "tinyint": pa.int8(),
    "double": pa.float64(),
    "datetime": pa.timestamp("s"),
    "varchar": pa.large_string(),
    "text": pa.large_string(),
    "mediumtext": pa.large_string(),
}


def arrow_type(mysql_data_type: str) -> pa.DataType:
    """Map a MySQL DATA_TYPE (as in information_schema.COLUMNS) to an Arrow type."""
    try:
        return _ARROW_TYPES[mysql_data_type.lower()]
    except KeyError:
        raise ValueError(f"No Arrow mapping for MySQL type {mysql_data_type!r}") from None


def clean_datetimes(values: list) -> tuple[list, int]:
    """Replace values PyMySQL could not parse as datetimes (e.g. '0000-00-00') with None.

    Returns the cleaned list and how many values were replaced.
    """
    cleaned = [v if v is None or isinstance(v, dt.datetime) else None for v in values]
    replaced = sum(1 for before, after in zip(values, cleaned) if before is not None and after is None)
    return cleaned, replaced


def table_schema(cursor, table: str) -> pa.Schema:
    cursor.execute(
        "SELECT COLUMN_NAME, DATA_TYPE FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s ORDER BY ORDINAL_POSITION",
        (table,),
    )
    columns = cursor.fetchall()
    if not columns:
        raise ValueError(f"Table {table!r} not found in the TAWOS database")
    return pa.schema([pa.field(name, arrow_type(data_type)) for name, data_type in columns])


def export_table(settings: dict, table: str, out_dir: Path, chunk_rows: int) -> dict:
    """Stream one table into <out_dir>/<table>.parquet and return its manifest entry."""
    with pymysql.connect(**settings, charset="utf8mb4") as meta:
        with meta.cursor() as cur:
            schema = table_schema(cur, table)
            cur.execute(f"SELECT COUNT(*) FROM `{table}`")
            expected_rows = cur.fetchone()[0]

    datetime_columns = {i for i, f in enumerate(schema) if pa.types.is_timestamp(f.type)}
    final_path = out_dir / f"{table}.parquet"
    tmp_path = out_dir / f"{table}.parquet.partial"
    rows = 0
    invalid_datetimes = 0

    # SSCursor streams rows from the server instead of loading the whole table into memory.
    with pymysql.connect(**settings, charset="utf8mb4", cursorclass=pymysql.cursors.SSCursor) as conn:
        with conn.cursor() as cur, pq.ParquetWriter(tmp_path, schema, compression="zstd") as writer:
            cur.execute(f"SELECT * FROM `{table}`")
            while batch := cur.fetchmany(chunk_rows):
                columns = [list(col) for col in zip(*batch)]
                for i in datetime_columns:
                    columns[i], replaced = clean_datetimes(columns[i])
                    invalid_datetimes += replaced
                arrays = [pa.array(col, type=field.type) for col, field in zip(columns, schema)]
                writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
                rows += len(batch)

    if rows != expected_rows:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"{table}: exported {rows} rows but MySQL reports {expected_rows}")
    tmp_path.replace(final_path)

    return {
        "file": final_path.name,
        "rows": rows,
        "columns": schema.names,
        "bytes": final_path.stat().st_size,
        "invalid_datetimes_set_to_null": invalid_datetimes,
    }


def sha256_of(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", nargs="+", default=TAWOS_TABLES, choices=TAWOS_TABLES, metavar="TABLE")
    parser.add_argument("--out", type=Path, default=config.TAWOS_PARQUET_DIR)
    parser.add_argument("--chunk-rows", type=int, default=50_000)
    args = parser.parse_args(argv)

    settings = config.tawos_mysql_settings()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"tables": {}}

    with pymysql.connect(**settings) as conn, conn.cursor() as cur:
        cur.execute("SELECT VERSION()")
        manifest["mysql_version"] = cur.fetchone()[0]

    if config.TAWOS_DUMP.exists():
        print(f"Hashing {config.TAWOS_DUMP.name} ...", flush=True)
        manifest["source_dump"] = {
            "file": config.TAWOS_DUMP.name,
            "bytes": config.TAWOS_DUMP.stat().st_size,
            "sha256": sha256_of(config.TAWOS_DUMP),
        }

    for table in args.tables:
        started = time.perf_counter()
        entry = export_table(settings, table, args.out, args.chunk_rows)
        entry["exported_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        manifest["tables"][table] = entry
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"{table:<17} {entry['rows']:>10,} rows  {entry['bytes'] / 1e6:>8.1f} MB  "
              f"{time.perf_counter() - started:>6.1f}s", flush=True)

    print(f"Wrote {len(args.tables)} table(s) to {args.out}")


if __name__ == "__main__":
    main()
