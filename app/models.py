from pydantic import BaseModel
from typing import List, Optional


class DocumentInfo(BaseModel):
    id: str
    name: str
    type: str
    uploaded_at: str
    chunk_count: int


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    content: str
    doc_name: str
    chunk_index: int
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]


class UploadResponse(BaseModel):
    success: bool
    doc_id: Optional[str] = None
    message: str
    chunk_count: int = 0


class DeleteResponse(BaseModel):
    success: bool
    message: str
