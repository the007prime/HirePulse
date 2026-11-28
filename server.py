"""HirePulse FastAPI Entry Point.

This file serves as the main server entry point for running uvicorn (`uvicorn server:app`)
and provides backward compatibility with existing tests and scripts.
"""

from app.main import app
from app.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE
from app.api.schemas import ChatRequest, MemoryCreateRequest
from app.api.routes.threads import get_thread_title as _thread_title

__all__ = [
    "app",
    "ALLOWED_EXTENSIONS",
    "MAX_FILE_SIZE",
    "ChatRequest",
    "MemoryCreateRequest",
    "_thread_title",
]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
