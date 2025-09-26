FROM python:3.10-slim

# system deps: ffmpeg for clipping; libgl/glib for decord
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install deps (PyTorch CPU wheels first)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
      torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install --no-cache-dir -r /app/requirements.txt

# copy project
COPY . /app

# make repo root importable so config _base_ includes resolve
ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE 8000

# healthcheck (optional)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT','8000') + '/docs')" || exit 1

# start API (change to inference_api2:app if that's your file)
CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
