
# Tablokhan — Persian & Arabic OCR

A lightweight OCR system for detecting and recognizing Persian and Arabic text from images.

Tablokhan combines a PP-OCRv6-based text detector with PP-OCRv5 Mobile text recognition, using ONNX Runtime as the common inference backend.

The project provides two application interfaces built on the same OCR core:

- **Local:** FastAPI REST API + HTML frontend
- **Hugging Face:** Gradio + ONNX

The OCR pipeline, model configuration, reading order, cropping strategy, and recognition logic are shared between the two deployments.

---

## Overview

Tablokhan is designed for Arabic-script OCR, with particular attention to Persian and Arabic reading order.

The system processes an input image through the following pipeline:

```text
Input Image
      │
      ▼
Detection Preprocessing
(LAB CLAHE + Resize + Normalization)
      │
      ▼
PP-OCRv6 Medium FT
ONNX Text Detection
      │
      ▼
DB Post-Processing
      │
      ▼
Quadrilateral Text Boxes
      │
      ▼
Persian / Arabic Reading Order
(top → bottom, right → left)
      │
      ▼
Perspective Polygon Crop
(CROP_MARGIN=15)
      │
      ▼
PP-OCRv5 Mobile
ONNX Text Recognition
      │
      ▼
CTC Decoding
      │
      ▼
Remove Empty Results
      │
      ▼
Multiline OCR Text + Detection Metadata
````

---

## Features

* Persian and Arabic text recognition
* PP-OCRv6 Medium FT text detection
* PP-OCRv5 Mobile text recognition
* ONNX Runtime inference
* Shared OCR core for local and Hugging Face deployments
* Polygon-based perspective cropping
* Persian/Arabic right-to-left reading order
* Top-to-bottom line ordering
* Multiline text output
* Detection and recognition confidence scores
* Automatic CUDA / CPU execution-provider selection
* FastAPI REST API for local deployment
* Browser-based local frontend
* Gradio interface for Hugging Face Spaces

---

## Architecture

The project separates the OCR engine from the application interfaces.

```text
                    ┌─────────────────────────┐
                    │   Shared ONNX OCR Core   │
                    │                         │
                    │  ONNXOCRService         │
                    │  ├─ Detection           │
                    │  ├─ DB PostProcess      │
                    │  ├─ Reading Order       │
                    │  ├─ Perspective Crop    │
                    │  └─ Recognition         │
                    └────────────┬────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
                 ▼                               ▼
       ┌──────────────────┐            ┌──────────────────┐
       │  Local Deployment │            │ Hugging Face     │
       │                  │            │                  │
       │ FastAPI          │            │ Gradio           │
       │ HTML Frontend    │            │ ZeroGPU-compatible│
       └──────────────────┘            └──────────────────┘
```

The central OCR implementation is located in:

```text
backend/services/onnx_ocr_service.py
```

This service is reused by both the local FastAPI application and the Hugging Face Gradio application. 

---

# Models

## Text Detection

**Model:** PP-OCRv6 Medium FT

**Format:** ONNX

The detector is responsible for locating text regions in the input image and produces quadrilateral polygons for detected text.

## Text Recognition

**Model:** PP-OCRv5 Mobile Recognition

**Format:** ONNX

The recognizer processes each cropped text region and performs CTC-based decoding using the character dictionary defined in `inference.yml`.

Recognition input shape:

```text
[1, 3, 48, 320]
```

The decoder is configured for Arabic/Persian text processing with reverse decoding enabled. 

---

# Detection Pipeline

Detection preprocessing uses:

### CLAHE

Contrast enhancement is applied to the **L channel of LAB color space**.

```text
clipLimit = 2.0
tileGridSize = (8, 8)
```

### Resize

The image is resized so that the longest side is limited to:

```text
640 pixels
```

The resulting dimensions are rounded to multiples of 32.

### Normalization

The detector uses RGB ImageNet normalization:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

These preprocessing steps are implemented directly in the shared ONNX OCR service. 

---

# Detection Configuration

The current detection configuration is:

```python
DET_THRESH = 0.30
DET_BOX_THRESH = 0.40
DET_UNCLIP_RATIO = 1.40
DET_RESIZE_LONG = 640
PADDING = 0
```

DB post-processing uses quadrilateral boxes:

```python
DBPostProcess(
    thresh=0.30,
    box_thresh=0.40,
    max_candidates=3000,
    unclip_ratio=1.40,
)
```

The final post-processing mode is:

```text
box_type = "quad"
```



---

# Reading Order

Persian and Arabic require a right-to-left reading order.

Tablokhan does not simply sort detected boxes by their X coordinate.

Instead, `reading_order.py` groups boxes into text lines using vertical interval containment and then applies:

```text
1. Top → Bottom between lines
2. Right → Left within the same line
```

This is implemented using a union-find grouping strategy followed by line sorting. 

This step is particularly important for multi-region images containing Persian or Arabic text.

---

# Text Cropping

Each detected quadrilateral is converted into a rectified perspective crop before recognition.

The crop process:

```text
Detected Polygon
      │
      ▼
Perspective Transformation
      │
      ▼
Rectified Text Crop
      │
      ▼
Border Extension
      │
      ▼
Recognition
```

The current target configuration uses:

```python
CROP_MARGIN = 15
```

The crop itself is performed with OpenCV perspective transformation and replicated borders. 

---

# Recognition Pipeline

For each detected text region:

1. A perspective crop is generated.
2. The crop is passed to the PP-OCRv5 Mobile recognition preprocessor.
3. The recognition ONNX model performs inference.
4. CTC decoding converts model output into text.
5. Arabic/Persian reverse decoding is enabled.
6. Empty recognition results are discarded.

The final recognized regions are then joined into multiline text:

```python
"\n".join(recognized_texts)
```



---

# Output

The OCR service returns both the final text and per-region metadata.

Example:

```json
{
  "success": true,
  "processing_time_ms": 812.4,
  "image_width": 1280,
  "image_height": 720,
  "text": "متن اول\nمتن دوم\nمتن سوم",
  "boxes": [
    {
      "polygon": [
        [100, 120],
        [320, 120],
        [320, 165],
        [100, 165]
      ],
      "text": "متن اول",
      "confidence": 0.9621,
      "det_score": 0.9417
    }
  ],
  "execution_provider": {
    "detection": "CPUExecutionProvider",
    "recognition": "CPUExecutionProvider"
  }
}
```

The actual API includes:

* final OCR text
* processing time
* original image dimensions
* recognized polygons
* recognition confidence
* detection score
* active ONNX Runtime execution providers



---

# Local Deployment

## 1. Clone the repository

```bash
git clone https://github.com/amre-sen/Tablokhan.git
cd Tablokhan
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

The local application uses FastAPI, Uvicorn, ONNX Runtime, OpenCV, PaddleX processing components, and the required OCR utilities. 

## 3. Download model files

The ONNX models are intentionally distributed through Hugging Face rather than committed directly to the GitHub repository.

Run:

```bash
python download_models.py
```

The script downloads:

```text
models/
├── detection/
│   └── PP-OCRv6_medium_FT.onnx
│
└── recognition/
    ├── inference.onnx
    └── inference.yml
```



## 4. Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or:

```bash
python main.py
```



## 5. Open the local interface

```text
http://127.0.0.1:8000
```

The local FastAPI application serves the HTML frontend directly from:

```text
frontend/index.html
```

The frontend supports image upload and sends OCR requests to the backend. 

---

# REST API

## OCR

```http
POST /ocr
```

Upload an image as a multipart form-data file.

Example with `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/ocr" \
  -F "file=@image.jpg"
```

## Status

```http
GET /status
```

The status endpoint reports:

* engine readiness
* detection execution provider
* recognition execution provider
* active OCR configuration



---

# Execution Providers

The ONNX service automatically checks available ONNX Runtime providers.

Preferred order:

```text
CUDAExecutionProvider
        ↓
CPUExecutionProvider
```

When a compatible CUDA environment is available, ONNX Runtime attempts to use CUDA.

Otherwise, the service automatically falls back to CPU execution. 

TensorRT is **not required by the shared ONNX architecture**.

---

# Hugging Face Deployment

A Gradio version of the same ONNX OCR pipeline is deployed on Hugging Face Spaces:

**[https://huggingface.co/spaces/amre-sen/Tablokhan](https://huggingface.co/spaces/amre-sen/Tablokhan)**

The Hugging Face application uses:

```text
Gradio
   +
ONNX Detection
   +
ONNX Recognition
```

The application reuses the same OCR service instead of maintaining a separate OCR implementation. 

The difference between the two deployments is the application layer:

```text
Local
FastAPI + HTML Frontend
        │
        ▼
Shared ONNX OCR Core

Hugging Face
Gradio
        │
        ▼
Shared ONNX OCR Core
```

This keeps model behavior and preprocessing consistent across environments.

---

# Project Structure

```text
Tablokhan/
│
├── backend/
│   ├── routes.py
│   └── services/
│       ├── onnx_ocr_service.py
│       └── ocr_service.py
│
├── frontend/
│   └── index.html
│
├── models/
│   ├── detection/
│   │   └── PP-OCRv6_medium_FT.onnx
│   │
│   └── recognition/
│       ├── inference.onnx
│       └── inference.yml
│
├── reading_order.py
├── main.py
├── app.py
├── config.py
├── download_models.py
├── requirements.txt
└── README.md
```

---

# Configuration

Model paths, OCR parameters, recognition dimensions, and server settings are centralized in:

```text
config.py
```

The configuration supports environment-variable overrides.

Examples:

```bash
OCR_DET_THRESH
OCR_DET_BOX_THRESH
OCR_DET_UNCLIP_RATIO
OCR_DET_RESIZE_LONG
OCR_CROP_MARGIN
OCR_REC_HEIGHT
OCR_REC_WIDTH
OCR_MODELS_DIR
```



---

# Design Principles

Tablokhan follows several practical design decisions:

### One OCR Core

Detection, cropping, reading order, and recognition are implemented once in the shared service.

### Separate Application Layers

The OCR engine is independent from the interface:

```text
OCR Core
   ├── FastAPI + HTML
   └── Gradio
```

### Models Outside Git History

Large model binaries are downloaded from Hugging Face rather than unnecessarily storing them in Git history.

### Model Initialization Once

The local FastAPI application initializes the OCR service at startup so model sessions are reused across requests instead of being recreated for every image. 

---

# Current OCR Configuration

| Component              | Configuration               |
| ---------------------- | --------------------------- |
| Detection              | PP-OCRv6 Medium FT          |
| Detection format       | ONNX                        |
| Recognition            | PP-OCRv5 Mobile             |
| Recognition format     | ONNX                        |
| Runtime                | ONNX Runtime                |
| Detection threshold    | `0.30`                      |
| Box threshold          | `0.40`                      |
| Unclip ratio           | `1.40`                      |
| Detection max side     | `640`                       |
| Detection padding      | `0`                         |
| CLAHE                  | LAB L-channel               |
| CLAHE clip limit       | `2.0`                       |
| CLAHE grid             | `(8, 8)`                    |
| Recognition input      | `48 × 320`                  |
| Crop margin            | `15`                        |
| Reading order          | Top → Bottom / Right → Left |
| Detection boxes        | Quadrilateral               |
| Recognition decoder    | CTC                         |
| Arabic/Persian reverse | Enabled                     |

---

# Limitations

Tablokhan is an OCR system focused on Persian and Arabic text in image inputs.

Performance can vary depending on:

* image resolution
* text size
* background complexity
* image quality
* text orientation
* detection quality
* hardware and ONNX Runtime provider

The reported processing time is environment-dependent and should not be interpreted as a fixed benchmark.

---

# Repository & Demo

GitHub:

**[https://github.com/amre-sen/Tablokhan](https://github.com/amre-sen/Tablokhan)**

Hugging Face:

**[https://huggingface.co/spaces/amre-sen/Tablokhan](https://huggingface.co/spaces/amre-sen/Tablokhan)**

---

# License

Apache License 2.0

