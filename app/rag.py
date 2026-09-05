import os
import json
import uuid
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from sentence_transformers import CrossEncoder

# 設定（可透過環境變數覆蓋）
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
CHROMA_DIR = os.environ.get("CHROMA_DIR", "/app/data/chroma")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/app/data/uploads")
DOCS_JSON = os.path.join(DATA_DIR, "documents.json")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))

os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 初始化 Embedding（使用 fastembed，不需要 PyTorch）
print("載入 embedding 模型 (fastembed)...")
embedding_fn = DefaultEmbeddingFunction()
print("embedding 模型載入完成")

# 初始化 Rerank 模型
print("載入 rerank 模型 (ms-marco-MiniLM-L-6-v2)...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("rerank 模型載入完成")

# 初始化 ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

# 文字分塊器（優先按段落分割）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "],
    keep_separator=True
)


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


def upload_document(file_path: str, file_name: str, file_type: str) -> dict:
    doc_id = str(uuid.uuid4())[:8]

    text = _parse_file(file_path, file_type)
    if not text.strip():
        return {"success": False, "message": "文件內容為空", "doc_id": None, "chunk_count": 0}

    chunks = text_splitter.split_text(text)
    if not chunks:
        return {"success": False, "message": "無法分割文件內容", "doc_id": None, "chunk_count": 0}

    # 存入 ChromaDB（ChromaDB 自動用 embedding function 計算向量）
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

    # 先多取一些候選結果供 rerank
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
    pairs = [[query, r["content"]] for r in search_results]
    scores = reranker.predict(pairs)

    for i, score in enumerate(scores):
        search_results[i]["score"] = round(float(score), 4)

    # 按 rerank 分數排序
    search_results.sort(key=lambda x: x["score"], reverse=True)

    # 去重：移除內容重複的 chunks（保留分數最高的）
    seen = set()
    unique_results = []
    for r in search_results:
        # 用前 100 字元做去重 key
        key = r["content"][:100]
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    return unique_results[:top_k]
