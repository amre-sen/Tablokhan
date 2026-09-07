"""Local PC entry point for Tablokhan OCR application.
Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000
or simply:
    python main.py
"""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from backend.routes import _get_service, router as ocr_router
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tablokhan_main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle manager.
    Loads ONNX models, DBPostProcess, and recognition processors ONCE
    at application startup so they are not reloaded on every inference request.
    """
    logger.info("Initializing Tablokhan ONNX OCR Service at startup...")
    try:
        service = _get_service()
        logger.info(
            "Tablokhan OCR Service loaded successfully! "
            "Detection Provider: %s | Recognition Provider: %s",
            service.det_provider,
            service.rec_provider,
        )
    except Exception as exc:
        logger.warning(
            "Model files not found or failed to load at startup: %s. "
            "Models will attempt initialization on the first request if added.",
            exc,
        )
    yield
    logger.info("Shutting down Tablokhan OCR Service.")


app = FastAPI(
    title="Tablokhan Local OCR",
    description="Local PC OCR service using FastAPI + ONNX Detection + ONNX Recognition for Arabic and Persian",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local client compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include OCR API router (/ocr, /status)
app.include_router(ocr_router)

# Serve frontend index.html at root '/'
frontend_dir = Path(__file__).resolve().parent / "frontend"
if (frontend_dir / "index.html").exists():

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(frontend_dir / "index.html")

    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

if __name__ == "__main__":
    logger.info("Starting local server at http://%s:%d", config.HOST, config.PORT)
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=False)
