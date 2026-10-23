# Built from the repository root (see docker-compose.yml): the service needs the prediction engine in
# ml-engine/ and the trained models in ml-engine/models/arena-v1, not only backend/.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /service
COPY ml-engine/pyproject.toml ml-engine/pyproject.toml
COPY ml-engine/src ml-engine/src

# By default without torch: every configuration the router picks (FastText / TF-IDF) works, while the SBERT ones,
# M3, the stack and DistilBERT are listed as "not installed". WITH_TORCH=true serves those too, without putting
# PyTorch in the image: this stage converts SBERT and the fine-tuned DistilBERT to ONNX (erp-export-onnx, which
# fails the build unless ONNX Runtime reproduces PyTorch's outputs), and the image runs them with ONNX Runtime and
# M3 in NumPy. The DistilBERT weights are not in git: a build on the machine that fine-tuned them exports them;
# anywhere else that configuration is listed as "not installed". The image keeps no PyTorch weight files.
# pip's and Hugging Face's downloads stay in BuildKit caches, not in the image, so a rebuild reuses them.
FROM base AS models
ARG WITH_TORCH=false
COPY ml-engine/models/arena-v1 /models/arena-v1
RUN --mount=type=cache,target=/root/.cache/pip --mount=type=cache,target=/root/.cache/huggingface \
    if [ "$WITH_TORCH" = "true" ]; then \
        pip install torch --index-url https://download.pytorch.org/whl/cpu && \
        pip install "./ml-engine[serving,onnx-export]" && \
        OMP_NUM_THREADS=1 erp-export-onnx /models/arena-v1; \
    fi && \
    rm -f /models/arena-v1/distilbert/*.safetensors

# The image: the prediction engine with the learners the arena's configurations use (XGBoost without its GPU
# library), then the service.
FROM base
ARG WITH_TORCH=false
RUN --mount=type=cache,target=/root/.cache/pip if [ "$WITH_TORCH" = "true" ]; then \
        pip install "./ml-engine[serving,serving-onnx]"; \
    else \
        pip install "./ml-engine[serving]"; \
    fi

COPY backend/pyproject.toml backend/pyproject.toml
COPY backend/app backend/app
RUN --mount=type=cache,target=/root/.cache/pip pip install ./backend

# The trained models, with the ONNX exports of the networks.
COPY --from=models /models/arena-v1 models/arena-v1

# NFR1 asks for 2 s at the 95th percentile with 10 concurrent users: one process runs the Python work of
# requests one at a time, so several worker processes share the load. Each worker holds its own models: ~270 MB
# without torch, so 4 workers (p95 0.53 s). With every configuration a worker that has used them all held ~1.6 GB
# in PyTorch (DistilBERT alone 545 MB), and 4 of them outgrew the 4 GB limit (NFR12), so 2 workers (p95 0.92 s).
# WEB_CONCURRENCY set at run time still wins.
# One native thread per worker (LightGBM's OpenMP, OpenBLAS, ONNX Runtime): with a thread per core in every worker
# the threads fought over the CPU, and p95 was 1.7 s instead of 0.5 s.
ENV MODELS_DIR=/service/models/arena-v1 \
    HF_HUB_OFFLINE=1 \
    WITH_TORCH=$WITH_TORCH \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /service/backend
EXPOSE 8004
CMD ["sh", "-c", "export WEB_CONCURRENCY=${WEB_CONCURRENCY:-$([ \"$WITH_TORCH\" = true ] && echo 2 || echo 4)}; exec uvicorn app.main:app --host 0.0.0.0 --port 8004"]
