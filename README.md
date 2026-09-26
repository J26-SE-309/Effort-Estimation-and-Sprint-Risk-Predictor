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
│   │   ├── main.py      # app setup, model loading at start-up, /health
│   │   ├── config.py    # settings from environment variables or backend/.env
│   │   ├── db.py        # this component's own PostgreSQL database
│   │   ├── tables.py    # prediction audit log, feedback, outcomes, pinned configurations
│   │   ├── store.py     # reading and writing them (predictions never wait for the database)
│   │   ├── prediction.py  # the link to the prediction engine in ml-engine
│   │   ├── schemas.py   # request and response models (proposal Appendix C)
│   │   └── api/v1/      # /estimate, /risk, /recommend, /compare, /models, pins, feedback, outcomes
│   ├── tests/
│   └── Dockerfile
├── ml-engine/           # Python package `erp`: data pipeline, labels, features, models
│   ├── src/erp/
│   │   ├── extract/     # TAWOS MySQL -> Parquet
│   │   ├── explore/     # data profiling
│   │   ├── snapshot/    # sprint timelines and the point-in-time snapshot
│   │   ├── labels/      # risk rules R1-R6 and the 200-story review sample
│   │   ├── features/    # the feature catalogue and its computation
│   │   ├── models/      # encoders, learners, M3, calibration (C1, C2), explanations, metrics
│   │   ├── arena/       # Phase 4: the Comparative Model Arena, stack, leaderboard, H1 / H2
│   │   ├── serving/     # Phase 5: live features, router (R1), explanations (X1), recommendations (R2),
│   │   │                #          sprint simulation (A1): the engine behind the API
│   │   └── tawos.py     # loading the exported tables, decoding TAWOS fields
│   ├── models/          # trained models, committed (LightGBM text, JSON, safetensors, skops; no pickles)
│   ├── reports/         # generated reports (e.g. tawos-profile.md)
│   └── tests/
└── docker-compose.yml   # effort-db + effort-api, and tawos-mysql for the one-off TAWOS import
```

## Getting Started

Requires Python 3.12, Docker Desktop and Git. Run these in PowerShell from the repository root after cloning.

1. Create a virtual environment and install the backend and ML engine with their test tools:

   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\python -m pip install -e "ml-engine[dev,serving]" -e "backend[dev]"
   ```

2. Start this component's local database (skip it when `backend/.env` names the hosted one; see
   [Database](#database)), then run the API with auto-reload:

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

To run the service and its database together in Docker instead: `docker compose up --build`. The image is
built from the repository root, because it holds the prediction engine and the trained models. By default it
leaves out torch and serves the FastText and TF-IDF configurations, which are all the router picks; the SBERT
ones, M3 and the stack show as "not installed" in `/api/v1/models`. To serve them too, build with torch (CPU
build, XGBoost, CatBoost and the SBERT model, about 1.5 GB more): set `EFFORT_WITH_TORCH=true` before
`docker compose up --build`. Locally, the same configurations need `pip install -e "ml-engine[serving-sbert]"`
after the CPU build of torch.

### Database

This component has its own PostgreSQL database and account. No other component connects to it. It holds the
prediction audit log (FR21: IDs, the model and version, the feature values and the answer; no story text),
feedback and outcomes (FR19) and pinned configurations (FR12).

- **Hosted (the real data): Neon**, region Singapore. Put its direct connection string (host without
  `-pooler`) in `backend/.env` as `DATABASE_URL=postgresql://...`, as Neon gives it (see `backend/.env.example`).
  `.env` is never committed. Both `uvicorn` and `docker compose up` then use it.
- **Local (development): the `effort-db` container**, used when `backend/.env` has no `DATABASE_URL`:

  | Setting | Local value |
  |---|---|
  | Host and port | `localhost:5444` |
  | Database | `effort_db` |
  | User / password | `effort_user` / `effort-local` |

- **Tests** use an in-memory SQLite database and never touch either.

The service applies the migrations in `backend/app/migrations` (Alembic) at start-up. After changing
`app/tables.py`, add a migration against the local database: `cd backend; alembic revision --autogenerate -m
"what changed"`, then check the generated file.

Neon's free plan keeps 6 hours of history, and the feedback and outcomes are evaluation data, so back them up
regularly: `cd backend; python -m app.backup` writes every table to
`Datasets/effort-risk/backups/effort-db-<time>.jsonl.gz`, and `python -m app.backup --restore FILE` restores one
into an empty database. The health check does not query a hosted database: the web app polls health every 15 s,
which would otherwise keep Neon from ever suspending (100 compute-hours a month on the free plan).

### API contract

The service answers with the Comparative Model Arena's trained models (`ml-engine/models/arena-v1`), loaded
once at start-up.

| Endpoint | What it does |
|---|---|
| `POST /api/v1/estimate` | Effort, interval, risk, confidence, reasons and recommendations for every story (Appendix C) |
| `POST /api/v1/risk` | The same, plus the sprint-level risk: Monte Carlo of committed effort against capacity (FR16) |
| `POST /api/v1/recommend` | Only the recommendations, per story and for the sprint (FR15) |
| `POST /api/v1/compare` | The same backlog through several configurations side by side (FR12) |
| `GET /api/v1/models` | The arena's configurations with their leaderboard metrics (FR10) |
| `GET / PUT / DELETE /api/v1/projects/{id}/pin` | Pin a configuration for a project, overriding the router (FR12) |
| `POST /api/v1/feedback`, `POST /api/v1/outcomes` | A product owner's decision; what really happened (FR19) |
| `GET / POST /api/v1/projects/{id}/history` | The team's sprint history, and importing it (see [Sprint history](#sprint-history)) |

NFR1 load test against a running service (10 users, each sending 50-story backlogs back to back):
`python backend/loadtest.py`. In the container (`docker compose up`: 4 worker processes with one native thread
each, `WEB_CONCURRENCY` and `OMP_NUM_THREADS`), the 95th percentile was 0.53 s over 200 requests, using about
900 MB of memory; one 50-story request alone takes 0.16 s. With the hosted database it was 0.56 s: the audit log
is written after the response is sent and pins are cached for 10 s, so predictions do not wait for the database.
Every prediction is recorded, so the load test refuses to run against a service using the hosted database; run
it with the local one. Explanations cost the most for TF-IDF + SVR / SVM
(about 2 s per 50 stories, feature-group occlusion over an expensive kernel model); the router never picks it
for a live project, but a pinned SVR / SVM will miss NFR1.

The router (R1) answers with the arena's pooled winner unless a project's own winner is clearly better or a
configuration is pinned; every prediction names its configuration, model version and the feature groups it used,
and is stored in the audit log (FR21). Requests may add the team's recent delivery (`team_context`) and the
sprint (`sprint_context`); otherwise they come from the project's sprint history, and without either the
prediction is flagged as a cold start and its confidence is lower.
The shared JSON Schemas live in
[`Synapse-Web/contracts/effort-estimation`](https://github.com/J26-SE-309/Synapse-Web/tree/main/contracts/effort-estimation);
keep them in sync with `backend/app/schemas.py`.

### Sprint history

The models learned from each team's past sprints: velocity (points finished in the last 3 closed sprints), how
much it varies, how often stories spill over or are reopened, and how long stories take (FR5). **The platform owns
the sprints.** Until it serves them through an API, this service keeps copies of the sprint records in its own
tables, each labelled with where it came from: `imported` (a CSV through the API), `tawos` (real TAWOS sprints,
development data) or `synthetic` (made up for tests and demos, never used in any evaluation). `SOURCE` in
`backend/app/history.py` is the one place that says where records come from: when the platform's API exists, a
client returning the same record format replaces it, and the development data is deleted.

| Endpoint | What it does |
|---|---|
| `GET /api/v1/projects/{id}/history` | What the models see about the team now, whether it is still a cold start, and its sprints |
| `POST /api/v1/projects/{id}/history` | Import the project's sprint history from a CSV (`text/csv`); replaces its records |

Every estimate fills the team and sprint context from the project's history; values the caller sends win, and
`feature_sources` says which answered (`history` or `request`). After 3 closed sprints a team is no longer a cold
start. The numbers are computed by `ml-engine/src/erp/serving/history.py` with the training pipeline's own code
(`erp/features/team.py`); `erp-history-parity` replays the real TAWOS sprints through it and compares with the
features training used ([`history-parity.md`](ml-engine/reports/history-parity.md)): every feature matches for
all 20,827 stories, except the cycle time of 31 stories, where stories resolved in the same second meet at the
edge of "the last 50".

**The record format (the CSV import):** one row per story per sprint (a story carried into the next sprint has a
row in each), with a header. A row without a `sprint_id` is a resolved story that was never in a sprint: only its
cycle time counts. Times are ISO 8601 (UTC when no zone is given). Example:
[`backend/examples/sprint-history-synthetic.csv`](backend/examples/sprint-history-synthetic.csv).

| Column | Meaning |
|---|---|
| `sprint_id`, `sprint_name` | The sprint (required for a story in a sprint: `sprint_started_at`, `sprint_planned_end`, `committed_at`) |
| `sprint_started_at`, `sprint_planned_end`, `sprint_closed_at` | When the sprint started, was planned to end, and closed (empty while it runs) |
| `story_id`, `issue_type` | The story (required); Story, Task, Bug, Improvement or New Feature |
| `committed_at`, `left_at` | When it was committed to the sprint; when it was taken out before the end (empty: still in) |
| `points_at_commit`, `points_at_close` | Its story points then |
| `done_in_sprint` | `true` if finished in this sprint (the sprint's velocity) |
| `spilled_over`, `reopened` | At the story's first sprint: not done by the end (risk rule R1); reopened after done, then or in the next sprint (R6) |
| `started_at`, `resolved_at`, `hours_in_progress` | When work started, when it was resolved, and the hours it spent in progress (cycle time) |

The import checks every row (dates in order, one value per sprint, no negative points, each story once per sprint)
and lists every problem by row; nothing is imported while there are any.

Development data, run from `backend/` (into the database `backend/.env` names, else the local one):

```powershell
python -m app.devdata load                    # TAWOS-MESOS and TAWOS-INDY (needs the Datasets folder) and the SYN-* teams
python -m app.devdata list
python -m app.devdata delete                  # all development records, and those projects' predictions
```

The `SYN-*` teams cover what TAWOS lacks: a brand-new team, teams with 1 and 2 closed sprints (still cold starts),
steady and erratic teams, and a team that never estimates. They are synthetic and must not be used to evaluate
the models (ML guide 7.2); name them in the proposal's AI-use disclosure (Appendix H).

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
│       ├── arena/arena-v1/      # encoded text per fold, predictions, inner-fold predictions, Optuna trials
│       ├── mlflow/              # MLflow experiment tracking (mlflow.db)
│       └── logs/                # logs of long runs
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

9. Compare the first models with the baselines (ML guide Phase 2): SBERT + LightGBM for effort (M1) and
   risk (M2) on a time-ordered split, with text-only and features-only versions. The first run downloads the
   SBERT model (about 90 MB, pinned to one revision) into the standard Hugging Face cache of your machine and
   caches every story's embedding in `Datasets/effort-risk/embeddings/`:

   ```powershell
   .venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
   .venv\Scripts\pip install -e "ml-engine[encoders]"
   .venv\Scripts\erp-train-first
   ```

10. Train the deployable bundle (Phase 3): M1 and M2 with the risk calibrator (C1), effort intervals (C2) and
    SHAP explanations (X1). It is saved to [`ml-engine/models/sbert-lightgbm-v1/`](ml-engine/models/) and
    committed: `m1.txt` and `m2.txt` are the LightGBM models, `bundle.json` holds everything else (encoder
    and revision, input columns, calibrator, interval quantiles, threshold, test metrics, code commit):

    ```powershell
    .venv\Scripts\erp-train-bundle
    ```

11. Run the Comparative Model Arena (Phase 4): every Appendix D configuration (TF-IDF + Random Forest,
    TF-IDF + SVR / SVM, FastText + LightGBM, SBERT + XGBoost / LightGBM / CatBoost, SBERT + multi-task MLP M3)
    tuned with Optuna on time-ordered folds inside the training split, trained, calibrated (C1, C2) and saved
    to [`ml-engine/models/arena-v1/`](ml-engine/models/). About three hours on a laptop CPU; `--configs`
    runs a subset, and a second process can train `tfidf-svm` (single-threaded) at the same time after
    `--prepare` has fitted the text encoders:

    ```powershell
    .venv\Scripts\pip install -e "ml-engine[encoders,arena]"
    .venv\Scripts\erp-train-arena --prepare
    .venv\Scripts\erp-train-arena
    ```

12. Fine-tune DistilBERT (E4) on a CUDA GPU in a separate environment with the CUDA build of torch, then
    finish it on the CPU (predictions, C1, C2):

    ```powershell
    python -m venv .venv-gpu
    .venv-gpu\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu130
    .venv-gpu\Scripts\pip install -e ml-engine transformers safetensors
    .venv-gpu\Scripts\python -m erp.arena.distilbert
    .venv\Scripts\python -m erp.arena.distilbert --finish
    ```

13. Build the stacked ensemble, the leaderboard the router reads (`ml-engine/models/arena-v1/leaderboard.json`)
    and the arena report, then run the H1 and H2 experiments. Training fits adaptive C2 intervals and the
    confidence score automatically; `erp-fit-intervals` refits them for every saved configuration and checks
    that the confidence bands order the accuracy:

    ```powershell
    .venv\Scripts\erp-fit-intervals
    .venv\Scripts\erp-arena-report
    .venv\Scripts\erp-run-hypotheses
    ```

Each step writes a report to [`ml-engine/reports/`](ml-engine/reports/):
[`tawos-profile.md`](ml-engine/reports/tawos-profile.md) (what TAWOS contains),
[`sprint-timeline.md`](ml-engine/reports/sprint-timeline.md) (sprint histories, the clock check behind their
tolerance, worked examples), [`snapshot.md`](ml-engine/reports/snapshot.md) (the filtering log and the
training rows), [`labels.md`](ml-engine/reports/labels.md) (how often each warning sign fires) and
[`features.md`](ml-engine/reports/features.md) (every feature, its meaning and why it was known at commitment),
[`first-models.md`](ml-engine/reports/first-models.md) (baselines and the first M1 and M2 results) and
[`uncertainty-and-explanations.md`](ml-engine/reports/uncertainty-and-explanations.md) (calibration, interval coverage and explanations of the bundle),
[`arena.md`](ml-engine/reports/arena.md) (the leaderboard, significance tests and NFR checks) and
[`hypotheses.md`](ml-engine/reports/hypotheses.md) (H1: joint learning; H2: the upstream quality signals) and
[`confidence.md`](ml-engine/reports/confidence.md) (adaptive intervals and the proposed confidence score).

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
