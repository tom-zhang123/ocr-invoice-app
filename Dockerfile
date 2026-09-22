FROM python:3.12-slim

# libgl1 / libglib2.0-0 是 opencv(rapidocr 依赖) 运行所需的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV RAPIDOCR_MODEL_ROOT_DIR=/opt/rapidocr-models

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build the image with all required model weights so runtime does not depend on the network.
COPY ocr_models.py .
RUN python -c "from ocr_models import warmup_models; warmup_models()"

COPY . .

EXPOSE 8000

# 单 worker 足够；CPU 密集识别，高并发可加 --workers 2
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
