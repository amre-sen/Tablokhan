````markdown
# Tablokhan — Persian & Arabic OCR

A lightweight OCR system for Persian and Arabic text detection and recognition.

The project uses **PP-OCRv6 Medium FT** for text detection and **PP-OCRv5 Mobile** for text recognition. Both models are exported to **ONNX** and run with ONNX Runtime in the local version.

## Features

- Persian and Arabic text OCR
- PP-OCRv6 Medium FT text detection
- PP-OCRv5 Mobile text recognition
- ONNX Runtime inference
- Polygon-based text cropping
- Persian/Arabic reading order
- Right-to-left ordering within the same text line
- Multiline text output
- FastAPI REST API for local use
- Gradio interface for the Hugging Face demo

## Pipeline

```text
Input Image
    ↓
PP-OCRv6 Medium FT
ONNX Text Detection
    ↓
DBPostProcess
    ↓
Reading Order
(top → bottom, right → left)
    ↓
Perspective Polygon Crop
    ↓
PP-OCRv5 Mobile
ONNX Text Recognition
    ↓
Remove Empty Results
    ↓
"\n".join(...)
    ↓
Multiline Text
````

## Detection Configuration

The current detection configuration is:

```python
DET_THRESH = 0.30
DET_BOX_THRESH = 0.40
DET_UNCLIP_RATIO = 1.40
DET_RESIZE_LONG = 640
PADDING = 0
```

Detection preprocessing includes CLAHE on the L channel in LAB color space:

```text
clipLimit = 2.0
tileGridSize = (8, 8)
```

## Recognition

Recognition uses:

```text
PP-OCRv5 Mobile
```

with:

```text
Input height: 48
Input width : 320
```

The recognition pipeline preserves the Persian/Arabic processing and CTC decoding used by the original model configuration.

## Reading Order

Detected text regions are ordered using the project's `reading_order.py`.

The reading order is:

1. Top to bottom
2. Right to left for regions belonging to the same line

This is important for Persian and Arabic text.

## Local Installation

Clone the repository:

```bash
git clone https://github.com/amre-sen/Tablokhan.git
cd Tablokhan
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Download the model files:

```bash
python download_models.py
```

The models are downloaded from Hugging Face rather than stored directly in this GitHub repository.

## Local Usage

Start the FastAPI application:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

The browser interface sends images to:

```text
POST /ocr
```

## API Response

The API returns a single multiline text field:

```json
{
  "success": true,
  "processing_time_ms": 812.4,
  "text": "متن اول\nمتن دوم\nمتن سوم"
}
```

Empty recognition results are removed before joining the final text.

## Model Files

The project expects:

```text
models/
├── detection/
│   └── PP-OCRv6_medium_FT.onnx
│
└── recognition/
    ├── inference.onnx
    └── inference.yml
```

Model files are not required to be committed to GitHub.
Run:

```bash
python download_models.py
```

to obtain them.

## Hugging Face Demo

A live Gradio version of the ONNX pipeline is available on Hugging Face Spaces:

**[https://huggingface.co/spaces/amre-sen/Tablokhan](https://huggingface.co/spaces/amre-sen/Tablokhan)**

The Hugging Face version uses:

```text
Gradio
+
ONNX Detection
+
ONNX Recognition
```

The OCR pipeline remains the same; only the application interface and hosting environment are different.

## Project Structure

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
│   └── recognition/
│
├── reading_order.py
├── main.py
├── app.py
├── config.py
├── download_models.py
├── requirements.txt
└── README.md
```

## Runtime

The local application can use:

```text
CUDAExecutionProvider
```

when a compatible NVIDIA GPU and ONNX Runtime GPU environment are available.

Otherwise it can run with:

```text
CPUExecutionProvider
```

The CPU version does not require TensorRT, but inference is slower.

## License

Apache License 2.0
