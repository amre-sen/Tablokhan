"""Configuration file for Tablokhan Local OCR.
Centralizes model paths, detection parameters, and server settings.
"""

import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# --- Model Paths (overridable via environment variables) -------------------
MODELS_DIR = os.getenv("OCR_MODELS_DIR", str(BASE_DIR / "models"))
DET_ONNX_PATH = os.getenv(
    "OCR_DET_ONNX_PATH",
    str(Path(MODELS_DIR) / "detection" / "PP-OCRv6_medium_FT.onnx"),
)
REC_ONNX_PATH = os.getenv(
    "OCR_REC_ONNX_PATH",
    str(Path(MODELS_DIR) / "recognition" / "inference.onnx"),
)
REC_YAML_PATH = os.getenv(
    "OCR_REC_YAML_PATH",
    str(Path(MODELS_DIR) / "recognition" / "inference.yml"),
)

# --- Detection Parameters (Strictly Preserved) ------------------------------
DET_THRESH = float(os.getenv("OCR_DET_THRESH", "0.30"))
DET_BOX_THRESH = float(os.getenv("OCR_DET_BOX_THRESH", "0.40"))
DET_UNCLIP_RATIO = float(os.getenv("OCR_DET_UNCLIP_RATIO", "1.40"))
DET_RESIZE_LONG = int(os.getenv("OCR_DET_RESIZE_LONG", "640"))
PADDING = int(os.getenv("OCR_PADDING", "0"))

# --- Crop Parameter ---------------------------------------------------------
CROP_MARGIN = int(os.getenv("OCR_CROP_MARGIN", "15"))

# --- Recognition Shape ------------------------------------------------------
REC_HEIGHT = int(os.getenv("OCR_REC_HEIGHT", "48"))
REC_WIDTH = int(os.getenv("OCR_REC_WIDTH", "320"))

# --- Server Configuration ---------------------------------------------------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
