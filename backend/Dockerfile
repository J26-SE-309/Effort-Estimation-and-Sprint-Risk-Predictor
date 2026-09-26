# Built from the repository root (see docker-compose.yml): the service needs the prediction engine in
# ml-engine/ and the trained models in ml-engine/models/arena-v1, not only backend/.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /service

# The prediction engine with the learners the arena's configurations use. WITH_TORCH=false leaves out torch and
# SBERT: the router's choices (FastText / TF-IDF configurations) still work, the SBERT and M3 ones do not.
ARG WITH_TORCH=true
COPY ml-engine/pyproject.toml ml-engine/pyproject.toml
COPY ml-engine/src ml-engine/src
RUN if [ "$WITH_TORCH" = "true" ]; then \
        pip install torch --index-url https://download.pytorch.org/whl/cpu && \
        pip install "./ml-engine[serving,encoders]" && \
        python -c "from sentence_transformers import SentenceTransformer; from erp.models.encoders import SbertEncoder as E; SentenceTransformer(E.model_id, revision=E.revision, device='cpu')"; \
    else \
        pip install "./ml-engine[serving]"; \
    fi

COPY backend/pyproject.toml backend/pyproject.toml
COPY backend/app backend/app
RUN pip install ./backend

# The trained models (the DistilBERT weights are not in git; without them that configuration is simply absent).
COPY ml-engine/models/arena-v1 models/arena-v1

# NFR1 asks for 2 s at the 95th percentile with 10 concurrent users: one process runs the Python work of
# requests one at a time, so several worker processes share the load (each holds its own models, ~270 MB).
ENV MODELS_DIR=/service/models/arena-v1 \
    HF_HUB_OFFLINE=1 \
    WEB_CONCURRENCY=4

WORKDIR /service/backend
EXPOSE 8004
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]
