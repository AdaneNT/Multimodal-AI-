# R2-Tuning FastAPI Wrapper 

This repository provides a **FastAPI + Swagger UI interface** for running inference with [R2-Tuning](https://github.com/yeliudev/R2-Tuning).

It lets you:
- Upload a video
- Provide a text query
- Auto-generate clips via ffmpeg
- Test everything via Swagger UI at `/docs`

---
## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/AdaneNT/mm-ai-clipping.git
cd mm-ai-clipping
```
###  2. Build Docker image
Portable CPU-only image (runs on Intel & Apple Silicon):
```
docker buildx build --platform linux/amd64 -t r2-tuning:cpu .
```
### 3. Run container
Model checkpoint is already included in ./checkpoints:
```docker run --rm -p 8000:8000 \
  -e PYTHONPATH=/app \
  -v "$(pwd)/checkpoints:/app/checkpoints:ro" \
  -v "$(pwd)/data:/app/data" \
  r2-tuning:cpu
```
### 4. Open Swagger UI
```http://localhost:8000/docs```

### Optional
Instead of manual docker run:

```docker compose up --build```
Then open:

```http://localhost:8000/docs```

## License Notice

This wrapper is **not the original R2-Tuning code**.  
The core model comes from [R2-Tuning (Ye Liu et al.)](https://github.com/yeliudev/R2-Tuning), which is licensed under the BSD 3-Clause License (see [THIRD_PARTY_LICENSES.md](./THIRD_PARTY_LICENSES.md)).

All credit for the model architecture and training belongs to the original authors.  
This repo only provides a **Dockerized inference API** for easier testing at own premises.

---

## 🚀 Quickstart

### 1. Clone repositories
```bash
# Clone original R2-Tuning
git clone https://github.com/AdaneNT/R2-Tuning.git

# Clone this wrapper
git clone https://github.com/<your-username>/r2-tuning-fastapi-wrapper.git
cd r2-tuning-fastapi-wrapper
