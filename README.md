# Tablokhan Local OCR (FastAPI + ONNX)

A local PC version of Tablokhan using FastAPI, ONNX Detection, and ONNX Recognition for Arabic and Persian.

## راهنمای اجرا (مخصوص پردازنده CPU)
اگر فقط CPU دارید، هیچ نیازی به نصب onnxruntime-gpu ندارید! پکیج پیش‌فرض requirements.txt دقیقاً برای شماست:

```bash
cd Tablokhan-local
pip install -r requirements.txt
python download_models.py
uvicorn main:app --host 0.0.0.0 --port 8000
```

مرورگر خود را باز کرده و به آدرس زیر بروید:
http://127.0.0.1:8000
