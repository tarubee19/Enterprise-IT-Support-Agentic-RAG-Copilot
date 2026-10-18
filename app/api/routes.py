import logging
import secrets
from pathlib import Path

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.rag.vectorstore import add_documents
from app.rag.workflow import ask
from app.services.audit import write_audit
from app.services.ingestion import SUPPORTED, chunk_documents, load_file

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
settings = get_settings()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
UNSAFE_ADMIN_KEYS = {"", "change-me", "change-me-in-production"}


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000)


def _check_admin_key(provided: str) -> None:
    expected = settings.admin_api_key
    if expected in UNSAFE_ADMIN_KEYS:
        raise HTTPException(status_code=503, detail="Uploads are disabled until ADMIN_API_KEY is set")
    if not secrets.compare_digest(provided.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid admin key")


@router.get("/health")
def health():
    return {"status": "ok", "service": settings.app_name}


@router.post("/chat")
def chat(payload: ChatRequest):
    try:
        result = ask(payload.question)
    except Exception:
        logger.exception("Agent run failed")
        raise HTTPException(status_code=500, detail="The assistant could not answer this question. Please try again.")
    write_audit(payload.question, result["source_used"], result.get("trace", []))
    return {
        "answer": result["answer"],
        "source_used": result["source_used"],
        "trace": result.get("trace", []),
        "citations": result.get("citations", []),
        "rewritten_query": result.get("current_query", payload.question),
    }


@router.post("/ingest")
async def ingest(file: UploadFile = File(...), x_admin_key: str = Header(default="")):
    _check_admin_key(x_admin_key)
    name = Path(file.filename or "").name
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Supported: {', '.join(sorted(SUPPORTED))}")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than 10 MB")
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / name
    dest.write_bytes(data)
    chunks = chunk_documents(load_file(dest))
    ids = add_documents(chunks)
    return {"message": "Document indexed", "file": dest.name, "chunks": len(chunks), "ids_created": len(ids)}
