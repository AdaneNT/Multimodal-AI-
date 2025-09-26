# R2-Tuning FastAPI Wrapper 🚀

This repository provides a **FastAPI + Swagger UI interface** for running inference with [R2-Tuning](https://github.com/yeliudev/R2-Tuning).

It lets you:
- Upload a video
- Provide a text query
- Get predicted highlight segments
- Auto-generate clips via ffmpeg
- Test everything via Swagger UI at `/docs`

---

## ⚠️ License Notice

This wrapper is **not the original R2-Tuning code**.  
The core model comes from [R2-Tuning (Ye Liu et al.)](https://github.com/AdaneNT/R2-Tuning), which is licensed under the BSD 3-Clause License (see [THIRD_PARTY_LICENSES.md](./THIRD_PARTY_LICENSES.md)).

All credit for the model architecture and training belongs to the original authors.  
This repo only provides a **Dockerized inference API** for easier testing.

---

## 🚀 Quickstart

### 1. Clone repositories
```bash
# Clone original R2-Tuning
git clone https://github.com/AdaneNT/R2-Tuning.git

# Clone this wrapper
git clone https://github.com/<your-username>/r2-tuning-fastapi-wrapper.git
cd r2-tuning-fastapi-wrapper
