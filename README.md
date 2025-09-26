# R2-Tuning FastAPI Wrapper 

This repository provides a **FastAPI + Swagger UI interface** for running inference with [R2-Tuning](https://github.com/yeliudev/R2-Tuning). It provides:
- A FastAPI wrapper that turns the original command-line inference script into an HTTP service with Swagger UI.
- A Dockerized environment for CPU-only execution (portable across Intel and Apple Silicon).
- Ready-to-use endpoints for uploading videos + text queries and retrieving highlight clips.

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

## License 

- Original R2-Tuning (Ye Liu, BSD-3-Clause).  
- This wrapper: MIT License.  
See [THIRD_PARTY_LICENSES.md]([THIRD_PARTY_LICENSES.md](https://github.com/yeliudev/R2-Tuning/blob/main/LICENSE)) for details.




