from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Document(BaseModel):
    id: str
    title: str
    content: str
    metadata: Optional[dict] = None

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    results: List[Document]

class UploadResponse(BaseModel):
    message: str
    document_id: str

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

class ChatRequest(BaseModel):
    question: str
    use_graph: bool = False
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, gt=0)

class SourceDocument(BaseModel):
    content: Optional[str] = Field(None, description="The text content of the source chunk.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata associated with the source chunk (e.g., doc_id, chunk_seq_id).")

class ChatResponse(BaseModel):
    answer: str = Field(description="The generated answer from the assistant.")
    sources: List[SourceDocument] = Field(default_factory=list, description="List of source document chunks used to generate the answer, including metadata.")

class BatchChatRequestItem(BaseModel):
    question: str
    use_graph: bool = False

class BatchChatRequest(BaseModel):
    questions: List[BatchChatRequestItem]
    document_title: Optional[str] = None

class BatchChatResponseItem(BaseModel):
    question: str
    use_graph: bool
    answer: Optional[str] = None
    error: Optional[str] = None
    sources: Optional[List[SourceDocument]] = None

class BatchChatResponse(BaseModel):
    results: List[BatchChatResponseItem]
