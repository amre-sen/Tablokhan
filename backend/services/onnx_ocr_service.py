"""ONNX OCR service for Tablokhan (Local PC & Hugging Face Spaces).
Detection : PP-OCRv6 medium FT (ONNX, via onnxruntime)
Recognition: PP-OCRv5 mobile rec (ONNX, via onnxruntime)

Pipeline:
  1. Detection Preprocessing: LAB CLAHE + 32-multiple resize + ImageNet norm
  2. ONNX Detection
  3. DBPostProcess (quad boxes, det_thresh=0.30, box_thresh=0.40, unclip_ratio=1.40)
  4. Reading order sorting (top-to-bottom, right-to-left within line via reading_order.py)
  5. Perspective polygon crop (with CROP_MARGIN=10)
  6. Recognition Preprocessing & ONNX Recognition (PP-OCRv5 Mobile)
  7. CTC Label Decode with Arabic/Persian reverse=True
  8. Final text: "\n".join(non-empty recognized strings)

NO TensorRT is used. CUDAExecutionProvider is preferred when available,
with automatic graceful fallback to CPUExecutionProvider.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import yaml

from paddlex.inference.models.text_detection.processors import DBPostProcess
from paddlex.inference.models.text_recognition.processors import (
    CTCLabelDecode,
    OCRReisizeNormImg,
)

root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from reading_order import sort_boxes_reading_order
except ImportError:
    from services.reading_order import sort_boxes_reading_order

logger = logging.getLogger("onnx_ocr_service")


def _create_session(model_path: str) -> Tuple[ort.InferenceSession, str]:
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        try:
            session = ort.InferenceSession(
                model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            actual = session.get_providers()[0]
            logger.info("Loaded %s with provider=%s", model_path, actual)
            return session, actual
        except Exception:
            logger.warning(
                "CUDAExecutionProvider failed for %s - falling back to CPU.", model_path
            )

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    actual = session.get_providers()[0] if session.get_providers() else "CPUExecutionProvider"
    logger.info("Loaded %s with provider=%s", model_path, actual)
    return session, actual


class ONNXOCRService:
    def __init__(
        self,
        det_onnx_path: str,
        rec_onnx_path: str,
        rec_yaml_path: str,
        det_thresh: float = 0.30,
        det_box_thresh: float = 0.40,
        det_unclip_ratio: float = 1.40,
        det_resize_long: int = 640,
        rec_height: int = 48,
        rec_width: int = 320,
        crop_margin: int = 10,
    ) -> None:
        for path in (det_onnx_path, rec_onnx_path, rec_yaml_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required model file not found: {path}")

        self.det_thresh = det_thresh
        self.det_box_thresh = det_box_thresh
        self.det_unclip_ratio = det_unclip_ratio
        self.det_resize_long = det_resize_long
        self.rec_height = rec_height
        self.rec_width = rec_width
        self.crop_margin = crop_margin

        # DBPostProcess
        self.postprocess = DBPostProcess(
            thresh=det_thresh,
            box_thresh=det_box_thresh,
            max_candidates=3000,
            unclip_ratio=det_unclip_ratio,
        )
        self.postprocess.box_type = "quad"

        # Recognition YAML + processors
        with open(rec_yaml_path, "r", encoding="utf-8") as f:
            rec_config = yaml.safe_load(f)
        character_list = rec_config["PostProcess"]["character_dict"]

        self.resizer = OCRReisizeNormImg(
            rec_image_shape=[3, self.rec_height, self.rec_width], input_shape=None
        )
        self.decoder = CTCLabelDecode(character_list=character_list, use_space_char=True)
        self.decoder.reverse = True

        # Sessions loaded ONCE
        self.det_session, self.det_provider = _create_session(det_onnx_path)
        self.det_input_name = self.det_session.get_inputs()[0].name
        self.det_output_name = self.det_session.get_outputs()[0].name

        self.rec_session, self.rec_provider = _create_session(rec_onnx_path)
        self.rec_input_name = self.rec_session.get_inputs()[0].name
        self.rec_output_name = self.rec_session.get_outputs()[0].name

    def _detection_preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, list]:
        # CLAHE on LAB L-channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        image = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        src_h, src_w = image.shape[:2]
        h, w = image.shape[:2]
        ratio = self.det_resize_long / max(h, w)
        if ratio < 1.0:
            resize_h = int(h * ratio)
            resize_w = int(w * ratio)
        else:
            resize_h = h
            resize_w = w

        resize_h = max(32, int(round(resize_h / 32) * 32))
        resize_w = max(32, int(round(resize_w / 32) * 32))
        resized = cv2.resize(image, (resize_w, resize_h))

        ratio_h = resize_h / float(src_h)
        ratio_w = resize_w / float(src_w)

        img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img, dtype=np.float32)

        shape_list = [src_h, src_w, ratio_h, ratio_w]
        return img, shape_list

    def _db_postprocess_quad(self, pred: np.ndarray, shape_list: list) -> List[Dict[str, Any]]:
        pred_map = np.asarray(pred, dtype=np.float32)
        if pred_map.ndim == 4:
            pred_map = pred_map[0, 0]
        elif pred_map.ndim == 3:
            pred_map = pred_map[0]

        bitmap = (pred_map > self.det_thresh).astype(np.uint8)
        src_h = float(shape_list[0])
        src_w = float(shape_list[1])
        boxes, scores = self.postprocess.boxes_from_bitmap(
            pred_map, bitmap, src_w, src_h, self.det_box_thresh, self.det_unclip_ratio
        )
        if boxes is None:
            return []

        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if boxes.size == 0:
            return []

        boxes = boxes.reshape(-1, 4, 2)
        valid: List[Dict[str, Any]] = []
        for box, score in zip(boxes, scores):
            score = float(score)
            if score < self.det_box_thresh or not np.isfinite(box).all():
                continue
            box[:, 0] = np.clip(box[:, 0], 0, src_w - 1)
            box[:, 1] = np.clip(box[:, 1], 0, src_h - 1)
            valid.append({"polygon": box.copy(), "score": score})
        return valid

    def _crop_polygon_perspective(
        self, image: np.ndarray, polygon: np.ndarray, margin: int
    ) -> Optional[np.ndarray]:
        pts = np.asarray(polygon, dtype=np.float32).reshape(4, 2)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)

        top_left = pts[np.argmin(s)]
        bottom_right = pts[np.argmax(s)]
        top_right = pts[np.argmin(diff)]
        bottom_left = pts[np.argmax(diff)]
        rect = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

        width_a = np.linalg.norm(rect[2] - rect[3])
        width_b = np.linalg.norm(rect[1] - rect[0])
        max_width = max(int(round(width_a)), int(round(width_b)))

        height_a = np.linalg.norm(rect[1] - rect[2])
        height_b = np.linalg.norm(rect[0] - rect[3])
        max_height = max(int(round(height_a)), int(round(height_b)))

        if max_width < 2 or max_height < 2:
            return None

        dst = np.array(
            [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        crop = cv2.warpPerspective(
            image, M, (max_width, max_height), borderMode=cv2.BORDER_REPLICATE
        )
        if margin > 0:
            crop = cv2.copyMakeBorder(crop, margin, margin, margin, margin, cv2.BORDER_REPLICATE)
        return crop

    def _recognize_crop(self, crop: Optional[np.ndarray]) -> Tuple[str, float]:
        if crop is None:
            return "", 0.0
        processed = self.resizer([crop.copy()])
        rec_input = processed[0]
        if rec_input.ndim == 3:
            rec_input = np.expand_dims(rec_input, axis=0)
        rec_input = np.ascontiguousarray(rec_input, dtype=np.float32)

        rec_outputs = self.rec_session.run(
            [self.rec_output_name], {self.rec_input_name: rec_input}
        )
        rec_texts, rec_scores = self.decoder([rec_outputs[0]])
        rec_text = rec_texts[0] if rec_texts else ""
        rec_score = float(rec_scores[0]) if rec_scores else 0.0
        return rec_text, rec_score

    def run(self, image: np.ndarray) -> Dict[str, Any]:
        start = time.perf_counter()
        src_h, src_w = image.shape[:2]

        det_input, shape_list = self._detection_preprocess(image)
        det_outputs = self.det_session.run([self.det_output_name], {self.det_input_name: det_input})
        pred = np.asarray(det_outputs[0], dtype=np.float32)
        boxes = self._db_postprocess_quad(pred, shape_list)

        # Reading order: top->bottom rows, right->left within line
        boxes = sort_boxes_reading_order(boxes)

        recognized_boxes: List[Dict[str, Any]] = []
        recognized_texts: List[str] = []
        for box in boxes:
            crop = self._crop_polygon_perspective(image, box["polygon"], margin=self.crop_margin)
            if crop is None:
                continue
            text, score = self._recognize_crop(crop)
            clean_text = text.strip() if text else ""
            if clean_text:
                recognized_texts.append(clean_text)
                poly_list = (
                    box["polygon"].tolist()
                    if isinstance(box["polygon"], np.ndarray)
                    else box["polygon"]
                )
                recognized_boxes.append(
                    {
                        "polygon": poly_list,
                        "text": clean_text,
                        "confidence": round(float(score), 4),
                        "det_score": round(float(box["score"]), 4),
                    }
                )

        final_text = "\n".join(recognized_texts)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "success": True,
            "processing_time_ms": round(elapsed_ms, 2),
            "image_width": int(src_w),
            "image_height": int(src_h),
            "text": final_text,
            "boxes": recognized_boxes,
            "execution_provider": {
                "detection": self.det_provider,
                "recognition": self.rec_provider,
            },
        }


_service_instance: Optional[ONNXOCRService] = None


def get_ocr_service(
    det_onnx_path: str,
    rec_onnx_path: str,
    rec_yaml_path: str,
    **kwargs: Any,
) -> ONNXOCRService:
    global _service_instance
    if _service_instance is None:
        _service_instance = ONNXOCRService(det_onnx_path, rec_onnx_path, rec_yaml_path, **kwargs)
    return _service_instance
