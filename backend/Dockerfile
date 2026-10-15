# Built from the repository root (see docker-compose.yml): the service needs the prediction engine in
# ml-engine/ and the trained models in ml-engine/models/arena-v1, not only backend/.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /service

# The prediction engine with the learners the arena's configurations use. By default without torch: every
# configuration the router picks (FastText / TF-IDF) works, while the SBERT ones, M3, the stack and DistilBERT are
# listed as "not installed". WITH_TORCH=true adds the CPU build of torch, XGBoost, CatBoost, the SBERT model and
# DistilBERT's tokenizer and configuration (~1.5 GB); the fine-tuned DistilBERT weights come with the models below.
# pip's downloads stay in a BuildKit cache, not in the image, so a rebuild after a code change reuses them.
ARG WITH_TORCH=false
COPY ml-engine/pyproject.toml ml-engine/pyproject.toml
COPY ml-engine/src ml-engine/src
RUN --mount=type=cache,target=/root/.cache/pip if [ "$WITH_TORCH" = "true" ]; then \
        pip install torch --index-url https://download.pytorch.org/whl/cpu && \
        pip install "./ml-engine[serving,serving-sbert]" && \
        python -c "from sentence_transformers import SentenceTransformer; from erp.models.encoders import SbertEncoder as E; SentenceTransformer(E.model_id, revision=E.revision, device='cpu')" && \
        python -c "from transformers import AutoConfig, AutoTokenizer; from erp.arena.distilbert import MODEL_ID, REVISION; AutoConfig.from_pretrained(MODEL_ID, revision=REVISION); AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)"; \
    else \
        pip install "./ml-engine[serving]"; \
    fi

COPY backend/pyproject.toml backend/pyproject.toml
COPY backend/app backend/app
RUN --mount=type=cache,target=/root/.cache/pip pip install ./backend

# The trained models. The DistilBERT weights are not in git: a build on the machine that fine-tuned them includes
# them (served when WITH_TORCH=true); anywhere else that configuration is listed as "not installed".
COPY ml-engine/models/arena-v1 models/arena-v1

# NFR1 asks for 2 s at the 95th percentile with 10 concurrent users: one process runs the Python work of
# requests one at a time, so several worker processes share the load. Each worker holds its own models: ~270 MB
# without torch, so 4 workers (p95 0.53 s). With torch a worker that has used every configuration holds ~1.6 GB
# (DistilBERT alone 545 MB), and 4 of them outgrew the 4 GB limit (NFR12), so 2 workers (p95 0.92 s).
# WEB_CONCURRENCY set at run time still wins.
# One native thread per worker (LightGBM's OpenMP, OpenBLAS, torch): with a thread per core in every worker the
# threads fought over the CPU, and p95 was 1.7 s instead of 0.5 s.
ENV MODELS_DIR=/service/models/arena-v1 \
    HF_HUB_OFFLINE=1 \
    WITH_TORCH=$WITH_TORCH \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /service/backend
EXPOSE 8004
CMD ["sh", "-c", "export WEB_CONCURRENCY=${WEB_CONCURRENCY:-$([ \"$WITH_TORCH\" = true ] && echo 2 || echo 4)}; exec uvicorn app.main:app --host 0.0.0.0 --port 8004"]
