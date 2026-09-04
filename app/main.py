import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import SearchRequest, SearchResponse, UploadResponse, DeleteResponse
from . import rag

app = FastAPI(title="RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/app/data/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(rag.get_documents())}


@app.get("/documents")
def list_documents():
    return {"documents": rag.get_documents()}


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    allowed_types = {
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "text/plain": "txt"
    }

    file_type = allowed_types.get(file.content_type)
    if not file_type:
        ext = os.path.splitext(file.filename)[1].lower()
        type_map = {".pdf": "pdf", ".docx": "docx", ".txt": "txt"}
        file_type = type_map.get(ext)

    if not file_type:
        raise HTTPException(status_code=400, detail="不支援的檔案格式，僅支援 PDF、DOCX、TXT")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = rag.upload_document(file_path, file.filename, file_type)

    if os.path.exists(file_path):
        os.remove(file_path)

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])

    return UploadResponse(**result)


@app.delete("/documents/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str):
    result = rag.delete_document(doc_id)
    return DeleteResponse(**result)


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    results = rag.search(request.query, request.top_k)
    return SearchResponse(results=[r for r in results])
