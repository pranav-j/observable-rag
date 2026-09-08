FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

# install dependencies first for layer caching
COPY pyproject.toml ./
COPY src ./src
RUN pip install -e .

# then the rest of the app
COPY . .

EXPOSE 8000
CMD ["uvicorn", "observable_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]