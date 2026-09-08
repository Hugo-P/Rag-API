FROM python:3.11-slim

ARG EMBEDDING_PROVIDER=local

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 只有 local 模式才安裝 torch + sentence-transformers
COPY requirements-local.txt .
RUN if [ "$EMBEDDING_PROVIDER" = "local" ]; then \
        echo "安裝本地 embedding 依賴..." && \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
        pip install --no-cache-dir -r requirements-local.txt; \
    else \
        echo "跳過本地 embedding 依賴 (provider=$EMBEDDING_PROVIDER)"; \
    fi

COPY app/ ./app/

RUN mkdir -p /app/data/chroma /app/data/uploads

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
