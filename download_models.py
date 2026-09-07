"""Utility script to download required ONNX models for Tablokhan Local OCR."""
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DET_DIR = BASE_DIR / "models" / "detection"
REC_DIR = BASE_DIR / "models" / "recognition"
DET_DIR.mkdir(parents=True, exist_ok=True)
REC_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    {
        "name": "PP-OCRv6_medium_FT.onnx",
        "path": DET_DIR / "PP-OCRv6_medium_FT.onnx",
        "url": "https://huggingface.co/spaces/amre-sen/Tablokhan/resolve/main/models/detection/PP-OCRv6_medium_FT.onnx",
    },
    {
        "name": "inference.onnx",
        "path": REC_DIR / "inference.onnx",
        "url": "https://huggingface.co/spaces/amre-sen/Tablokhan/resolve/main/models/recognition/inference.onnx",
    },
    {
        "name": "inference.yml",
        "path": REC_DIR / "inference.yml",
        "url": "https://huggingface.co/spaces/amre-sen/Tablokhan/raw/main/models/recognition/inference.yml",
    },
]

for item in FILES:
    target = item["path"]
    if target.exists() and target.stat().st_size > 1000:
        print(f"✓ {item['name']} already exists")
    else:
        print(f"Downloading {item['name']}...")
        urllib.request.urlretrieve(item["url"], target)
print("Model check complete!")
