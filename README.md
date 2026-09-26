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
│       └── interim/             # derived tables: sprint timelines, then the snapshot
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

The TAWOS findings that shape the labelling and feature pipeline are in
[`ml-engine/reports/tawos-profile.md`](ml-engine/reports/tawos-profile.md); the sprint timelines, the
clock check behind their tolerance and worked examples are in
[`ml-engine/reports/sprint-timeline.md`](ml-engine/reports/sprint-timeline.md).

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
