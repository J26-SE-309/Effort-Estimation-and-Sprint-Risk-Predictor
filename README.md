# Effort Estimation and Sprint Risk Predictor

> A microservice of **Synapse**, an AI-assisted agile project platform built by research group **J26-SE-309**.

![Status](https://img.shields.io/badge/status-initial%20setup-orange)

## Overview

This service estimates the effort needed for backlog items and predicts risks to sprint delivery, so teams can plan sprints with realistic commitments.

Scope, methodology, datasets and models will be documented here as the research progresses.

## Repository Layout

```
Effort-Estimation-and-Sprint-Risk-Predictor/
├── docker-compose.yml   # local MySQL, used once to restore the TAWOS dump
├── ml-engine/           # Python package `erp`: data pipeline, labels, features, models
│   ├── src/erp/
│   │   ├── extract/     # TAWOS MySQL -> Parquet
│   │   ├── explore/     # data profiling
│   │   └── tawos.py     # loading the exported tables, decoding TAWOS fields
│   ├── reports/         # generated reports (e.g. tawos-profile.md)
│   └── tests/
└── backend/             # FastAPI prediction service consumed by Synapse-Web (planned)
```

## Data

Data is never stored in this repository. Everything lives in a `Datasets` folder **next to** the repositories (override with the `ERP_DATA_DIR` environment variable):

```
AgilePlatform/
├── Datasets/
│   ├── TAWOS/TAWOS.sql          # TAWOS v1.1 MySQL dump, doi.org/10.5522/04/21308124 (4.1 GB)
│   └── effort-risk/
│       └── tawos-raw/           # every TAWOS table as Parquet, plus _manifest.json
└── Effort-Estimation-Sprint-Risk-Predictor/   # this repository
```

## Getting Started (Windows, PowerShell)

Requires Python 3.12 and Docker Desktop. Run these from the repository root.

1. Create a virtual environment **outside** the repository and install the ML engine:

   ```powershell
   py -3.12 -m venv ..\.venvs\effort-risk
   ..\.venvs\effort-risk\Scripts\python -m pip install -e "ml-engine[dev]"
   ```

   The repository has no `.gitignore` yet, so also keep Python's `__pycache__` folders out of it (they would otherwise be picked up by `git add .`):

   ```powershell
   $venv = (Resolve-Path ..\.venvs\effort-risk).Path
   Set-Content "$venv\Lib\site-packages\erp_pycache_prefix.pth" "import sys; sys.pycache_prefix = r'$venv\pycache'"
   ```

2. Restore TAWOS into MySQL (one time, about 5 minutes):

   ```powershell
   docker compose up -d tawos-mysql
   docker exec tawos-mysql sh -c "mysql -uroot -ptawos-local tawos < /import/TAWOS.sql"
   ```

3. Export every table to Parquet (about 3 minutes). After this, MySQL can be stopped with `docker compose stop`:

   ```powershell
   ..\.venvs\effort-risk\Scripts\erp-export-tawos
   ```

4. Profile the data and run the tests:

   ```powershell
   ..\.venvs\effort-risk\Scripts\erp-profile-tawos
   cd ml-engine; ..\..\.venvs\effort-risk\Scripts\python -m pytest
   ```

The TAWOS findings that shape the labelling and feature pipeline are in [`ml-engine/reports/tawos-profile.md`](ml-engine/reports/tawos-profile.md).

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
