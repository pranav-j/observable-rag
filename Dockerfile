# Self-contained image: app, models and a built index, so a started container
# needs nothing from the network except the LLM provider.
#
# The index is built HERE rather than at runtime. Qdrant runs embedded (no
# QDRANT_URL), which means one process holds an exclusive lock on the storage
# directory -- fine for a single container, and the reason this doesn't try to
# scale horizontally.
#
#   docker build --platform linux/arm64 -t observable-rag .
#
# Platform matters: containers are not architecture-neutral. An arm64 image
# won't run on an x86_64 host. Build for whatever the target instance is
# (t4g/Graviton is arm64).
FROM python:3.12-slim

# The docs revision the index is built from. Pinned, because chunk ids are the
# contract the golden set is written against.
ARG FASTAPI_REF=0.159.1
ENV FASTAPI_REF=${FASTAPI_REF}

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models \
    HF_HUB_DISABLE_TELEMETRY=1

# CPU-only torch, explicitly, BEFORE the package pulls the default wheel.
# PyPI's torch bundles the NVIDIA CUDA runtime -- 3.3GB of it -- which is dead
# weight on a Graviton instance with no GPU. Installing from PyTorch's cpu index
# first means the editable install below finds the requirement already met.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Dependencies next, so edits to the app don't reinstall any of it.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# Bake the embedder and the cross-encoder into the image. Without this the
# first request downloads ~180MB and the visitor waits for it.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
import observable_rag.index.vector as v, observable_rag.retrieve.rerank as r; \
SentenceTransformer(v.MODEL_NAME); CrossEncoder(r.RERANK_MODEL); \
print('models cached')"

COPY . .

# Fetch the docs and build both indexes at build time. Needs network here so it
# needs none later; a cold start opens files instead of embedding 953 chunks.
RUN python -m observable_rag.index.build && \
    python -m observable_rag.ingest.manifest --check

# Drop privileges. The app only ever reads its data directory.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app /opt/models
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4).status==200 else 1)"

# One worker on purpose: embedded Qdrant takes an exclusive lock on its storage,
# so a second worker would fail to start.
CMD ["uvicorn", "observable_rag.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
