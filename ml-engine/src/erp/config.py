"""Paths and connection settings shared by the pipeline.

Data never lives in the repository. Datasets, extracted tables and trained
models go under ERP_DATA_DIR, which defaults to the Datasets folder next to
the repositories (AgilePlatform/Datasets).
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = Path(os.environ.get("ERP_DATA_DIR", REPO_ROOT.parent / "Datasets")).resolve()

# Raw input: the TAWOS MySQL dump, restored once into the tawos-mysql container.
TAWOS_DUMP = DATA_DIR / "TAWOS" / "TAWOS.sql"

# Everything this component generates lives under one folder.
WORK_DIR = DATA_DIR / "effort-risk"
TAWOS_PARQUET_DIR = WORK_DIR / "tawos-raw"  # one Parquet file per TAWOS table, unmodified
INTERIM_DIR = WORK_DIR / "interim"  # derived tables (sprint timelines, snapshots) that later steps read

# Trained models are code-sized (LightGBM text files and JSON) and are committed with the code that made them.
MODEL_BUNDLES_DIR = REPO_ROOT / "ml-engine" / "models"


def tawos_mysql_settings() -> dict:
    """Connection settings for the local tawos-mysql container (see docker-compose.yml)."""
    return {
        "host": os.environ.get("TAWOS_MYSQL_HOST", "127.0.0.1"),
        "port": int(os.environ.get("TAWOS_MYSQL_PORT", "3306")),
        "user": os.environ.get("TAWOS_MYSQL_USER", "root"),
        "password": os.environ.get("TAWOS_MYSQL_PASSWORD", "tawos-local"),
        "database": os.environ.get("TAWOS_MYSQL_DATABASE", "tawos"),
    }
