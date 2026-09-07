"""FastAPI router for Tablokhan Local OCR.
Defines POST /ocr and system status endpoints.
"""

import logging
from typing import Any, Dict
import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile
import numpy as np

import config
from backend.services.onnx_ocr_service import get_ocr_service

logger = logging.getLogger("tablokhan_routes")
router = APIRouter()


def _get_service():
    """Retrieve or initialize the singleton OCR service."""
    return get_ocr_service(
        det_onnx_path=config.DET_ONNX_PATH,
        rec_onnx_path=config.REC_ONNX_PATH,
        rec_yaml_path=config.REC_YAML_PATH,
        det_thresh=config.DET_THRESH,
        det_box_thresh=config.DET_BOX_THRESH,
        det_unclip_ratio=config.DET_UNCLIP_RATIO,
        det_resize_long=config.DET_RESIZE_LONG,
        rec_height=config.REC_HEIGHT,
        rec_width=config.REC_WIDTH,
        crop_margin=config.CROP_MARGIN,
    )


@router.post("/ocr")
async def perform_ocr(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Process an uploaded image through the ONNX OCR pipeline.
    Pipeline:
        1. Decode uploaded file bytes to OpenCV BGR image.
        2. Detection with PP-OCRv6 medium FT.
        3. DBPostProcess with quad polygons.
        4. Reading order sorting (top-to-bottom, right-to-left within rows).
        5. Perspective polygon crop (CROP_MARGIN=10).
        6. PP-OCRv5 Mobile recognition with CTC decoding.
        7. Filter empty text and newline join.
    Returns:
        {
            "success": True,
            "processing_time_ms": 812.4,
            "text": "متن اول\nمتن دوم\nمتن سوم"
        }
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Supported formats include PNG, JPEG, BMP, WebP.",
        )

    try:
        service = _get_service()
        result = service.run(image)
        return {
            "success": result["success"],
            "processing_time_ms": result["processing_time_ms"],
            "image_width": result.get("image_width", image.shape[1]),
            "image_height": result.get("image_height", image.shape[0]),
            "text": result["text"],
            "boxes": result.get("boxes", []),
            "execution_provider": result.get("execution_provider"),
        }
    except Exception as exc:
        logger.exception("OCR execution failed")
        raise HTTPException(status_code=500, detail=f"OCR execution failed: {str(exc)}")


@router.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get OCR engine status and active execution providers."""
    try:
        service = _get_service()
        return {
            "status": "ready",
            "execution_provider": {
                "detection": service.det_provider,
                "recognition": service.rec_provider,
            },
            "config": {
                "det_thresh": config.DET_THRESH,
                "det_box_thresh": config.DET_BOX_THRESH,
                "det_unclip_ratio": config.DET_UNCLIP_RATIO,
                "det_resize_long": config.DET_RESIZE_LONG,
                "crop_margin": config.CROP_MARGIN,
            },
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
