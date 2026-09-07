"""Hugging Face Spaces entry point (Gradio SDK, ZeroGPU compatible).
Reuses the shared ONNX OCR service (same core used by the local FastAPI version).
"""

import logging
import os
import sys
import traceback
from pathlib import Path
import cv2
import gradio as gr
import numpy as np

root_dir = str(Path(__file__).resolve().parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from backend.services.onnx_ocr_service import get_ocr_service
except ImportError:
    from services.onnx_ocr_service import get_ocr_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hf_ocr_app")

try:
    import spaces
    GPU_DECORATOR = spaces.GPU
except Exception:
    def GPU_DECORATOR(fn):
        return fn

MODELS_DIR = os.getenv("OCR_MODELS_DIR", "models")
DET_ONNX_PATH = os.getenv("OCR_DET_ONNX_PATH", os.path.join(MODELS_DIR, "detection", "PP-OCRv6_medium_FT.onnx"))
REC_ONNX_PATH = os.getenv("OCR_REC_ONNX_PATH", os.path.join(MODELS_DIR, "recognition", "inference.onnx"))
REC_YAML_PATH = os.getenv("OCR_REC_YAML_PATH", os.path.join(MODELS_DIR, "recognition", "inference.yml"))

_service = get_ocr_service(
    det_onnx_path=DET_ONNX_PATH,
    rec_onnx_path=REC_ONNX_PATH,
    rec_yaml_path=REC_YAML_PATH,
)

@GPU_DECORATOR
def run_ocr(pil_image):
    if pil_image is None:
        return "لطفاً ابتدا یک تصویر آپلود کنید.", "-", "-"
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    result = _service.run(image)
    return result["text"], f"{result['processing_time_ms']} ms", f"{result['execution_provider']}"

with gr.Blocks(title="Arabic/Persian OCR (ONNX)") as demo:
    gr.Markdown("## Arabic & Persian OCR (Tablokhan)")
    image_input = gr.Image(type="pil", label="Input")
    run_btn = gr.Button("Run OCR", variant="primary")
    text_output = gr.Textbox(label="Output", lines=12)
    run_btn.click(fn=run_ocr, inputs=image_input, outputs=text_output)

if __name__ == "__main__":
    demo.queue().launch()
