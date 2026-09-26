# Effort Estimation and Sprint Risk Predictor

> A microservice of **Synapse**, an AI-assisted agile project platform built by research group **J26-SE-309**.

![Status](https://img.shields.io/badge/status-initial%20setup-orange)

## Overview

This service estimates the effort needed for backlog items and predicts risks to sprint delivery, so teams can plan sprints with realistic commitments.

Scope, methodology, datasets and models will be documented here as the research progresses.

## Repository Layout

```
Effort-Estimation-and-Sprint-Risk-Predictor/
├── backend/             # FastAPI prediction service the platform calls (port 8004)
│   ├── app/
│   │   ├── main.py      # app setup and /health
│   │   ├── config.py    # settings from environment variables or backend/.env
│   │   ├── db.py        # this component's own PostgreSQL database
│   │   ├── schemas.py   # request and response models (proposal Appendix C)
│   │   └── api/v1/      # /estimate, /risk, /recommend, /models, /compare
│   ├── tests/
│   └── Dockerfile
├── ml-engine/           # Python package `erp`: data pipeline, labels, features, models
│   ├── src/erp/
│   │   ├── extract/     # TAWOS MySQL -> Parquet
│   │   ├── explore/     # data profiling
│   │   └── tawos.py     # loading the exported tables, decoding TAWOS fields
│   ├── reports/         # generated reports (e.g. tawos-profile.md)
│   └── tests/
└── docker-compose.yml   # effort-db + effort-api, and tawos-mysql for the one-off TAWOS import
```

## Getting Started

Requires Python 3.12, Docker Desktop and Git. Run these in PowerShell from the repository root after cloning.

1. Create a virtual environment and install the backend and ML engine with their test tools:

   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\python -m pip install -e "backend[dev]" -e "ml-engine[dev]"
   ```

2. Start this component's database, then run the API with auto-reload:

   ```powershell
   docker compose up -d effort-db
   cd backend
   ..\.venv\Scripts\uvicorn app.main:app --reload --port 8004
   ```

   Open http://localhost:8004/docs for the interactive API documentation.

3. Run the tests and the linter:

   ```powershell
   cd backend; ..\.venv\Scripts\python -m pytest; ..\.venv\Scripts\ruff check .
   cd ..\ml-engine; ..\.venv\Scripts\python -m pytest
   ```

To run the service and its database together in Docker instead: `docker compose up --build`.

### Database

This component has its own PostgreSQL database and account. No other component connects to it.

| Setting | Local value |
|---|---|
| Host and port | `localhost:5444` |
| Database | `effort_db` |
| User / password | `effort_user` / `effort-local` |

### API contract

Until the models are trained, `/estimate` returns placeholder predictions marked `model_version: "stub"` and
`/risk`, `/recommend` and `/compare` answer `501 Not Implemented`, so the gateway and dashboard can already be
built against the real shapes. The shared JSON Schemas live in
[`Synapse-Web/contracts/effort-estimation`](https://github.com/J26-SE-309/Synapse-Web/tree/main/contracts/effort-estimation);
keep them in sync with `backend/app/schemas.py`.

## ML pipeline: TAWOS data

Data is never stored in this repository. It lives in a `Datasets` folder **next to** the repositories
(override with the `ERP_DATA_DIR` environment variable):

```
AgilePlatform/
├── Datasets/
│   ├── TAWOS/TAWOS.sql          # TAWOS v1.1 MySQL dump, doi.org/10.5522/04/21308124 (4.1 GB)
│   └── effort-risk/
│       ├── tawos-raw/           # every TAWOS table as Parquet, plus _manifest.json
│       ├── interim/             # derived tables: sprint timelines, snapshot, labels, features
│       ├── review/              # the 200-story label-check workbooks (not in git)
│       ├── embeddings/          # cached SBERT embeddings, keyed by a hash of the story text
│       └── models/              # downloaded encoders and trained models
└── Effort-Estimation-and-Sprint-Risk-Predictor/   # this repository
```

1. Restore TAWOS into MySQL (one time, about 5 minutes):

   ```powershell
   docker compose up -d tawos-mysql
   docker exec tawos-mysql sh -c "mysql -uroot -ptawos-local tawos < /import/TAWOS.sql"
   ```

2. Export every table to Parquet (about 3 minutes), then stop MySQL with `docker compose stop tawos-mysql`:

   ```powershell
   .venv\Scripts\erp-export-tawos
   ```

3. Profile the data:

   ```powershell
   .venv\Scripts\erp-profile-tawos
   ```

4. Rebuild every issue's sprint timeline from the change history (about a minute):

   ```powershell
   .venv\Scripts\erp-build-timeline
   ```

5. Build the point-in-time snapshot: one row per story, as it was when it was committed to its first sprint
   (needs step 4):

   ```powershell
   .venv\Scripts\erp-build-snapshot
   ```

6. Compute the risk labels R1–R6 for every snapshot story:

   ```powershell
   .venv\Scripts\erp-build-labels
   ```

7. Draw the 200 stories for the human label check and write one workbook per reviewer to
   `Datasets/effort-risk/review/` (the automatic labels stay in a separate key file):

   ```powershell
   .venv\Scripts\erp-review-sample
   ```

   It refuses to overwrite workbooks that were already handed out (`--force` draws a new sample).

8. Compute the features: every clue the models see about a story, as it was known at commitment
   (rerun after step 6 whenever the label rules change):

   ```powershell
   .venv\Scripts\erp-build-features
   ```

9. Train the first models (ML guide Phase 2): baselines, then SBERT + LightGBM for effort (M1) and risk (M2)
   on a time-ordered split. The first run downloads the SBERT model (about 90 MB) into
   `Datasets/effort-risk/models/` and caches every story's embedding in `Datasets/effort-risk/embeddings/`:

   ```powershell
   .venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
   .venv\Scripts\pip install -e "ml-engine[encoders]"
   .venv\Scripts\erp-train-first
   ```

Each step writes a report to [`ml-engine/reports/`](ml-engine/reports/):
[`tawos-profile.md`](ml-engine/reports/tawos-profile.md) (what TAWOS contains),
[`sprint-timeline.md`](ml-engine/reports/sprint-timeline.md) (sprint histories, the clock check behind their
tolerance, worked examples), [`snapshot.md`](ml-engine/reports/snapshot.md) (the filtering log and the
training rows), [`labels.md`](ml-engine/reports/labels.md) (how often each warning sign fires) and
[`features.md`](ml-engine/reports/features.md) (every feature, its meaning and why it was known at commitment) and
[`first-models.md`](ml-engine/reports/first-models.md) (baselines and the first M1 and M2 results).

## Synapse Platform Services

| Service | Repository | Type |
|---|---|---|
| Synapse Web | [Synapse-Web](https://github.com/J26-SE-309/Synapse-Web) | Frontend |
| **Effort Estimation and Sprint Risk Predictor** | [Effort-Estimation-and-Sprint-Risk-Predictor](https://github.com/J26-SE-309/Effort-Estimation-and-Sprint-Risk-Predictor) | Backend + ML engine |
| Requirement Quality and Ambiguity Analyzer | [Requirement-Quality-and-Ambiguity-Analyzer](https://github.com/J26-SE-309/Requirement-Quality-and-Ambiguity-Analyzer) | Backend + ML engine |
| Requirement Traceability Engine | [Requirement-Traceability-Engine](https://github.com/J26-SE-309/Requirement-Traceability-Engine) | Backend + ML engine |
| User Story Refinement and Acceptance Criteria Generator | [User-Story-Refinement-Acceptance-Criteria-Generator](https://github.com/J26-SE-309/User-Story-Refinement-Acceptance-Criteria-Generator) | Backend + ML engine |

## Maintainer

- [@Nikeshala22](https://github.com/Nikeshala22) — service owner and project lead
