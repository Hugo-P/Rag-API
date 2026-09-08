import os
import json
import uuid
import math
import time
from abc import ABC, abstractmethod
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
import chromadb
import httpx

# ── 設定（可透過環境變數覆蓋）──────────────────────────────
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
CHROMA_DIR = os.environ.get("CHROMA_DIR", "/app/data/chroma")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/app/data/uploads")
DOCS_JSON = os.path.join(DATA_DIR, "documents.json")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))
RERANK_MODEL = os.environ.get("RERANK_MODEL", "")
DEVICE = os.environ.get("DEVICE", "cpu")

# Embedding provider 設定
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "local")  # local / openai / cloudflare / custom
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "")
EMBEDDING_API_URL = os.environ.get("EMBEDDING_API_URL", "")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "moka-ai/m3e-base")

# LLM provider 設定
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "none")  # none / openai
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_API_URL = os.environ.get("LLM_API_URL", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1024"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
DEFAULT_SYSTEM_PROMPT = os.environ.get("DEFAULT_SYSTEM_PROMPT", "") or """你是一個知識庫助手。根據以下提供的參考資料回答使用者的問題。
如果參考資料中沒有相關資訊，請明確告知使用者你無法根據現有資料回答。
回答時請使用繁體中文，並盡量引用資料來源。"""

os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Embedding Provider 抽象層 ────────────────────────────────
class EmbeddingProvider(ABC):
    """所有 embedding provider 的基底類別"""

    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        """將文字列表轉為向量列表"""
        ...

    def __call__(self, input):
        """ChromaDB EmbeddingFunction 相容介面"""
        if isinstance(input, str):
            input = [input]
        return self.encode(input)


class LocalEmbeddingProvider(EmbeddingProvider):
    """本地 sentence-transformers 模型"""

    def __init__(self, model_name: str, device: str):
        print(f"載入本地 embedding 模型 ({model_name})...")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        print("本地 embedding 模型載入完成")

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI / 相容 API（Ollama、vLLM、Together AI 等）"""

    def __init__(self, api_key: str, api_url: str, model: str):
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY 不能為空")
        self.api_url = (api_url or "https://api.openai.com/v1").rstrip("/")
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        print(f"OpenAI embedding provider 已初始化 (model={model}, url={self.api_url})")

    def encode(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.api_url}/embeddings"
        payload = {"input": texts, "model": self.model}

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=payload, headers=self.headers)
            resp.raise_for_status()

        data = resp.json()
        # 按 index 排序確保順序正確
        embeddings = sorted(data["data"], key=lambda x: x["index"])
        return [e["embedding"] for e in embeddings]


class CloudflareEmbeddingProvider(EmbeddingProvider):
    """Cloudflare Workers AI"""

    def __init__(self, api_key: str, api_url: str, model: str):
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY 不能為空")
        # api_url 應為 https://api.cloudflare.com/client/v4/accounts/{account_id}
        self.api_url = (api_url or "").rstrip("/")
        if not self.api_url:
            raise ValueError("EMBEDDING_API_URL 需包含 Cloudflare account ID，例如 https://api.cloudflare.com/client/v4/accounts/YOUR_ACCOUNT_ID")
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        print(f"Cloudflare embedding provider 已初始化 (model={model})")

    def encode(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.api_url}/ai/run/{self.model}"
        results = []

        # Cloudflare API 一次只能處理一個 input，需逐條呼叫
        with httpx.Client(timeout=60.0) as client:
            for text in texts:
                payload = {"text": text}
                resp = client.post(url, json=payload, headers=self.headers)
                resp.raise_for_status()
                data = resp.json()
                results.append(data["result"]["data"][0])
                # 避免觸發限流
                time.sleep(0.05)

        return results


class CustomEmbeddingProvider(EmbeddingProvider):
    """自訂 HTTP API

    預期 API 格式：
    POST {api_url}
    Request:  {"input": ["text1", "text2"]}
    Response: {"embeddings": [[0.1, ...], [0.2, ...]]}
    """

    def __init__(self, api_key: str, api_url: str, model: str):
        if not api_url:
            raise ValueError("EMBEDDING_API_URL 不能為空")
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        print(f"Custom embedding provider 已初始化 (url={self.api_url}, model={model})")

    def encode(self, texts: list[str]) -> list[list[float]]:
        payload = {"input": texts, "model": self.model}

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(self.api_url, json=payload, headers=self.headers)
            resp.raise_for_status()

        data = resp.json()
        return data["embeddings"]


# ── 初始化 Embedding Provider ─────────────────────────────────
def _create_embedding_provider() -> EmbeddingProvider:
    provider = EMBEDDING_PROVIDER.lower()

    if provider == "local":
        return LocalEmbeddingProvider(EMBEDDING_MODEL, DEVICE)
    elif provider == "openai":
        return OpenAIEmbeddingProvider(EMBEDDING_API_KEY, EMBEDDING_API_URL, EMBEDDING_MODEL)
    elif provider == "cloudflare":
        return CloudflareEmbeddingProvider(EMBEDDING_API_KEY, EMBEDDING_API_URL, EMBEDDING_MODEL)
    elif provider == "custom":
        return CustomEmbeddingProvider(EMBEDDING_API_KEY, EMBEDDING_API_URL, EMBEDDING_MODEL)
    else:
        raise ValueError(f"不支援的 EMBEDDING_PROVIDER: {provider}，可選值：local / openai / cloudflare / custom")


embedding_provider = _create_embedding_provider()


# ── ChromaDB 初始化 ───────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# 取得或建立 collection
try:
    collection = chroma_client.get_collection(
        name="knowledge_base",
        embedding_function=embedding_provider,
    )
    print(f"已載入現有 collection，包含 {collection.count()} 筆紀錄")
except Exception:
    collection_kwargs = {
        "name": "knowledge_base",
        "embedding_function": embedding_provider,
        "metadata": {"hnsw:space": "cosine"},
    }
    collection = chroma_client.create_collection(**collection_kwargs)
    print("已建立新 collection")


# ── Rerank 模型（可選）───────────────────────────────────────
reranker = None
if RERANK_MODEL:
    from sentence_transformers import CrossEncoder
    print(f"載入 rerank 模型 ({RERANK_MODEL})...")
    reranker = CrossEncoder(RERANK_MODEL, device=DEVICE)
    print("rerank 模型載入完成")
else:
    print("未設定 RERANK_MODEL，跳過 rerank")


# ── 文字分塊器 ────────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "],
    keep_separator=True
)


# ── 文件管理 ──────────────────────────────────────────────────
def _load_documents() -> dict:
    if os.path.exists(DOCS_JSON):
        with open(DOCS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"documents": []}


def _save_documents(data: dict):
    with open(DOCS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _parse_file(file_path: str, file_type: str) -> str:
    if file_type == "pdf":
        import fitz
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    elif file_type == "docx":
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        return "\n".join([doc.page_content for doc in docs])
    elif file_type == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"不支援的文件類型: {file_type}")


# ── API 函數 ──────────────────────────────────────────────────
def upload_document(file_path: str, file_name: str, file_type: str) -> dict:
    doc_id = str(uuid.uuid4())[:8]

    text = _parse_file(file_path, file_type)
    if not text.strip():
        return {"success": False, "message": "文件內容為空", "doc_id": None, "chunk_count": 0}

    chunks = text_splitter.split_text(text)
    if not chunks:
        return {"success": False, "message": "無法分割文件內容", "doc_id": None, "chunk_count": 0}

    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {"doc_id": doc_id, "doc_name": file_name, "chunk_index": i}
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas
    )

    docs_data = _load_documents()
    docs_data["documents"].append({
        "id": doc_id,
        "name": file_name,
        "type": file_type,
        "uploaded_at": datetime.now().isoformat(),
        "chunk_count": len(chunks)
    })
    _save_documents(docs_data)

    return {
        "success": True,
        "doc_id": doc_id,
        "message": f"成功上傳 {file_name}，切成 {len(chunks)} 個片段",
        "chunk_count": len(chunks)
    }


def delete_document(doc_id: str) -> dict:
    results = collection.get(where={"doc_id": doc_id})
    if results and results["ids"]:
        collection.delete(ids=results["ids"])

    docs_data = _load_documents()
    docs_data["documents"] = [d for d in docs_data["documents"] if d["id"] != doc_id]
    _save_documents(docs_data)

    return {"success": True, "message": f"已刪除文件 {doc_id}"}


def get_documents() -> list:
    docs_data = _load_documents()
    return docs_data.get("documents", [])


def search(query: str, top_k: int = 5) -> list:
    if collection.count() == 0:
        return []

    fetch_k = min(top_k * 4, collection.count())

    results = collection.query(
        query_texts=[query],
        n_results=fetch_k
    )

    search_results = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            search_results.append({
                "content": doc,
                "doc_name": metadata.get("doc_name", "未知"),
                "chunk_index": metadata.get("chunk_index", 0),
            })

    if not search_results:
        return []

    # Rerank：用 cross-encoder 重新評分
    if reranker:
        pairs = [[query, r["content"]] for r in search_results]
        scores = reranker.predict(pairs)

        for i, score in enumerate(scores):
            search_results[i]["score"] = round(1 / (1 + math.exp(-float(score))), 4)

        search_results.sort(key=lambda x: x["score"], reverse=True)
    else:
        for i, score in enumerate(results["distances"][0]):
            search_results[i]["score"] = round(1 - float(score), 4)

    # 去重
    seen = set()
    unique_results = []
    for r in search_results:
        key = r["content"][:100]
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    return unique_results[:top_k]


# ── LLM Provider 抽象層 ──────────────────────────────────────
class LLMProvider(ABC):
    """所有 LLM provider 的基底類別"""

    @abstractmethod
    def chat(self, messages: list[dict], max_tokens: int, temperature: float) -> str:
        """發送對話請求，回傳 LLM 回覆文字"""
        ...


class OpenAILLMProvider(LLMProvider):
    """OpenAI / 相容 API"""

    def __init__(self, api_key: str, api_url: str, model: str):
        if not api_key:
            raise ValueError("LLM_API_KEY 不能為空")
        self.api_url = (api_url or "https://api.openai.com/v1").rstrip("/")
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        print(f"OpenAI LLM provider 已初始化 (model={model}, url={self.api_url})")

    def chat(self, messages: list[dict], max_tokens: int, temperature: float) -> str:
        url = f"{self.api_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload, headers=self.headers)
            resp.raise_for_status()

        raw = resp.text

        # 嘗試解析 JSON，容錯處理多餘資料
        try:
            data = resp.json()
        except Exception:
            # 嘗試取第一個完整 JSON 物件
            import json as _json
            depth = 0
            end = 0
            for i, ch in enumerate(raw):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            data = _json.loads(raw[:end])

        return data["choices"][0]["message"]["content"]


class NoLLMProvider(LLMProvider):
    """不使用 LLM，直接回傳搜尋結果"""

    def chat(self, messages: list[dict], max_tokens: int, temperature: float) -> str:
        # 取最後一條 user 訊息作為 query
        query = ""
        for m in reversed(messages):
            if m["role"] == "user":
                query = m["content"]
                break

        results = search(query, top_k=5)
        if not results:
            return "找不到相關資料。"

        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[{i}] {r['content'][:300]}...")
        return "\n\n".join(parts)


# ── 初始化 LLM Provider ──────────────────────────────────────
def _create_llm_provider() -> LLMProvider:
    provider = LLM_PROVIDER.lower()

    if provider == "none":
        print("LLM 已停用 (LLM_PROVIDER=none)")
        return NoLLMProvider()
    elif provider == "openai":
        return OpenAILLMProvider(LLM_API_KEY, LLM_API_URL, LLM_MODEL)
    else:
        raise ValueError(f"不支援的 LLM_PROVIDER: {provider}，可選值：none / openai")


llm_provider = _create_llm_provider()


# ── Chat 函數（RAG + LLM）────────────────────────────────────
def chat(query: str, top_k: int = 5, system_prompt: str = "") -> dict:
    """RAG 流程：搜尋相關文件 → 組合 prompt → LLM 生成回答"""

    # 1. 搜尋相關文件
    results = search(query, top_k=top_k)

    if not results:
        return {
            "answer": "找不到相關資料，無法回答此問題。",
            "sources": [],
        }

    # 2. 組合 context
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(f"[{i}] (來源：{r['doc_name']}) {r['content']}")
    context = "\n\n".join(context_parts)

    # 3. 組合 prompt
    prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"參考資料：\n{context}\n\n問題：{query}"},
    ]

    # 4. LLM 生成回答
    answer = llm_provider.chat(messages, LLM_MAX_TOKENS, LLM_TEMPERATURE)

    return {
        "answer": answer,
        "sources": results,
    }
