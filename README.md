# OCR PDF WITH TESSERACT AND PADDLE OCR

## Setup and Install Environment
### Install installer **Tesseract**
- [Link here to download Tesseract Installer for Windows](https://github.com/UB-Mannheim/tesseract/wiki)
- Then download installers `tesseract-ocr-w64-setup-5.5.0.20241111.exe (64 bit)`

### Package Dependencies
```
python -m pip install pdf2image pytesseract opencv-python python-dotenv zipfile36
```
### PaddlePaddle 
```
# CPU
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
# GPU
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```
### PaddleOCR
```
python -m pip install paddleocr
```