# RAG API

通用 RAG（Retrieval-Augmented Generation）服務，基於 LangChain + ChromaDB + sentence-transformers。

## 功能

- 支援 PDF、Word (.docx)、純文字 (.txt) 上傳
- 自動解析文件 → 切塊 → 向量化 → 存入 ChromaDB
- 向量語意搜尋（cosine similarity）
- CrossEncoder Rerank 重排序，分數經 Sigmoid 歸一化到 0~1
- 搜尋結果自動去重
- REST API，任何語言都能呼叫
- 支援自訂 Embedding 和 Rerank 模型（透過環境變數）

## 技術棧

- **LangChain** — 流程編排
- **ChromaDB** — 向量資料庫
- **sentence-transformers** — Embedding + Rerank 模型
- **FastAPI** — REST API

## 快速開始

### Docker Compose（推薦）

```bash
docker-compose up -d --build
```

### 單獨 Docker

```bash
docker build -t rag-api .
docker run -p 8000:8000 -v ./data:/app/data rag-api
```

## API 端點

| 方法 | 路徑 | 功能 |
|------|------|------|
| GET | `/health` | 健康檢查 |
| GET | `/documents` | 文件列表 |
| POST | `/upload` | 上傳文件（multipart/form-data） |
| DELETE | `/documents/{doc_id}` | 刪除文件 |
| POST | `/search` | 向量搜尋 |

### 範例

**上傳文件：**
```bash
curl -X POST http://localhost:8000/upload -F "file=@document.pdf"
```

**搜尋：**
```bash
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"query":"如何退款","top_k":5}'
```

**文件列表：**
```bash
curl http://localhost:8000/documents
```

## 環境變數

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `EMBEDDING_MODEL` | `moka-ai/m3e-base` | Embedding 模型名稱（sentence-transformers 支援的模型） |
| `RERANK_MODEL` | （空） | Rerank 模型名稱（留空則不使用 rerank，用原始 cosine similarity） |
| `DEVICE` | `cpu` | 運算裝置（cpu / cuda） |
| `CHROMA_DIR` | `/app/data/chroma` | ChromaDB 儲存路徑 |
| `UPLOAD_DIR` | `/app/data/uploads` | 暫存上傳檔案路徑 |
| `CHUNK_SIZE` | `500` | 文字分塊大小（字元） |
| `CHUNK_OVERLAP` | `100` | 分塊重疊字元數 |
| `TOP_K` | `5` | 預設搜尋結果數量 |

> **注意：** 更換 `EMBEDDING_MODEL` 後，必須清空 `CHROMA_DIR` 並重新上傳所有文件，因為不同模型的向量維度不同。

### 常用中文模型

**Embedding：**
- `moka-ai/m3e-base` — 中文模型（預設）
- `BAAI/bge-small-zh-v1.5` — 輕量中文模型
- `BAAI/bge-base-zh-v1.5` — 中文模型（較大，效果更好）
- `BAAI/bge-large-zh-v1.5` — 中文模型（最大）

**Rerank：**
- `BAAI/bge-reranker-v2-m3` — 多語言 reranker（MaxKB 同款）
- `cross-encoder/ms-marco-MiniLM-L-6-v2` — 英文模型

## Docker Registry

### 推送到 Docker Hub
```bash
docker build -t yourusername/rag-api .
docker push yourusername/rag-api:latest
```

### 推送到 GitHub Container Registry
```bash
docker build -t ghcr.io/yourusername/rag-api .
docker push ghcr.io/yourusername/rag-api:latest
```

### docker-compose

```yaml
version: '3.8'
services:
  rag-api:
    container_name: rag-api
    image: ghcr.io/hugo-p/rag-api:latest
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - EMBEDDING_MODEL=moka-ai/m3e-base
      - DEVICE=cpu
      - CHROMA_DIR=/app/data/chroma
      - UPLOAD_DIR=/app/data/uploads
      - CHUNK_SIZE=500
      - CHUNK_OVERLAP=100
      - TOP_K=5
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
```
