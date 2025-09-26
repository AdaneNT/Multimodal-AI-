# Copyright (c) Ye Liu. Licensed under the BSD 3-Clause License.#
# This file is a modified version of the original R2-Tuning inference script.
# Changes: added FastAPI + Swagger UI wrapper and Dockerized deployment support.

from pathlib import Path
from typing import List, Optional, Tuple
import os
import re
import subprocess

import torch
import numpy as np
import torchvision.transforms.functional as F
import decord
from decord import VideoReader
import clip
import nncore
from nncore.engine import load_checkpoint
from nncore.nn import build_model

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ----------------------------- Defaults --------------------------------- #
DEFAULT_CONFIG = './configs/config_qvhighlights.py'
DEFAULT_CHECKPOINT = str(Path("./checkpoints/model_qvhighlights.pth"))

DATA_DIR = Path('./data').resolve()
UPLOAD_DIR = DATA_DIR / 'uploads'
CLIPS_DIR = DATA_DIR / 'clips'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------- FastAPI app ------------------------------- #
app = FastAPI(
    title=" Multimodal AI for Video Clipping",
    version="1.0.0",
    description="Upload a video and a text query to get relevant timestamps "
                "(start, end, score). Also auto-saves ONE clip for the best segment.",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},  # hides schemas section in Swagger
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------- Model state ------------------------------ #
_model = None
_cfg = None
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_model(config_path: str = DEFAULT_CONFIG, checkpoint: str = DEFAULT_CHECKPOINT):
    global _model, _cfg
    try:
        cfg = nncore.Config.from_file(config_path)
        cfg.model.init = True
        ckpt_path = checkpoint
        if ckpt_path.startswith('http'):
            ckpt_path = nncore.download(ckpt_path, out_dir='checkpoints')
        model = build_model(cfg.model, dist=False).eval().to(_device)
        model = load_checkpoint(model, ckpt_path, warning=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    _cfg = cfg
    _model = model


def _ensure_model():
    if _model is None or _cfg is None:
        _load_model()

# ----------------------------- Clip helpers ----------------------------- #
def _safe(s: str) -> str:
    """Make a filesystem-safe slug from a query string."""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    return s[:80] or "query"

def _ffmpeg_cut(input_path: str, start: float, end: float, out_path: str) -> str:
    """
    Cut [start, end] from input_path into out_path.
    Tries fast stream-copy first; if that fails (keyframe issues), re-encodes.
    Returns the actual path written.
    """
    start_s, end_s = f"{start:.2f}", f"{end:.2f}"

    cmd_copy = ["ffmpeg", "-y", "-ss", start_s, "-to", end_s, "-i", input_path, "-c", "copy", out_path]
    r = subprocess.run(cmd_copy, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode == 0 and Path(out_path).exists():
        return out_path

    out_path_re = str(Path(out_path).with_suffix(".reencode.mp4"))
    cmd_re = [
        "ffmpeg", "-y",
        "-ss", start_s, "-to", end_s,
        "-i", input_path,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        out_path_re
    ]
    r2 = subprocess.run(cmd_re, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r2.returncode == 0 and Path(out_path_re).exists():
        return out_path_re

    raise RuntimeError("ffmpeg failed to cut clip")

# ----------------------------- Video utils ------------------------------ #
def _load_video_tensor(video_path: str, cfg) -> Tuple[torch.Tensor, float]:
    decord.bridge.set_bridge('torch')
    vr = VideoReader(video_path)
    stride = vr.get_avg_fps() / cfg.data.val.fps
    fm_idx = [min(round(i), len(vr) - 1) for i in np.arange(0, len(vr), stride).tolist()]
    video = vr.get_batch(fm_idx).permute(0, 3, 1, 2).float() / 255

    size = 336 if '336px' in cfg.model.arch else 224
    h, w = video.size(-2), video.size(-1)
    s = min(h, w)
    x, y = round((h - s) / 2), round((w - s) / 2)
    video = video[..., x:x + s, y:y + s]
    video = F.resize(video, size=(size, size))
    video = F.normalize(video, (0.481, 0.459, 0.408), (0.269, 0.261, 0.276))
    video = video.reshape(video.size(0), -1).unsqueeze(0)

    return video, float(cfg.data.val.fps)

def _run_inference(video_path: str, query_text: str) -> List[Tuple[float, float, float]]:
    _ensure_model()
    device = next(_model.parameters()).device

    video, fps = _load_video_tensor(video_path, _cfg)
    tokens = clip.tokenize(query_text, truncate=True)

    with torch.inference_mode():
        pred = _model(dict(video=video.to(device), query=tokens.to(device), fps=[fps]))

    bnds = pred['_out']['boundary'].tolist()
    max_time = video.size(1) / fps
    results: List[Tuple[float, float, float]] = []
    for start, end, score in bnds:
        s = max(0.0, min(float(start), max_time))
        e = max(0.0, min(float(end), max_time))
        if e > s:
            results.append((round(s, 2), round(e, 2), float(score)))
    return results

# ----------------------------- Schemas ---------------------------------- #
class Segment(BaseModel):
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    score: float = Field(..., description="Confidence score")

class PredictResponse(BaseModel):
    query: str
    topk: int
    segments: List[Segment]
    saved_clip: Optional[str] = Field(
        None, description="Filesystem path to the saved best-clip (if created)"
    )

#class ReloadBody(BaseModel):
    #config_path: str = Field(DEFAULT_CONFIG, description="Path to config .py file")
    #checkpoint: str = Field(DEFAULT_CHECKPOINT, description="Local path or URL to checkpoint")

# ----------------------------- Endpoints -------------------------------- #
@app.on_event("startup")
def _startup():
    # Lazy-load on first request; uncomment to load immediately on startup:
    # _load_model()
    pass

@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Run text-guided highlight prediction",
    description=
        "Upload a video file " "and a text query. Returns top-K timestamp segments with scores, and saves ONE clip "
        "for the highest-score segment.",
    tags=["Video Clipping using Text Queries. "]
    
)
async def predict(
    response: Response,
    video: UploadFile = File(..., title="Input video", description="Upload a video file (mp4, mov, etc.)"),
    query: str = Form(..., description="Text query"),
    topk: int = Form(5, ge=1, le=50, description="Number of segments to return"),
):
    """ upload a file or provide a  path"""
    if video is None and not video_path:
        raise HTTPException(status_code=400, detail="Provide either 'video' upload or 'video_path'.")
 
    # Save uploaded file to disk
    if video is not None:
        suffix = Path(video.filename).suffix or '.mp4'
        save_path = UPLOAD_DIR / f"upload_{os.getpid()}_{video.filename}"
        with open(save_path, 'wb') as f:
            f.write(await video.read())
        path = str(save_path)
    else:
        path = video_path

    if not Path(path).exists():
        raise HTTPException(status_code=400, detail=f"Video path not found: {path}")

    try:
        results = _run_inference(path, query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # Keep top-k
    results = results[:topk]

    # Auto-save ONE clip for the highest score segment
    saved_clip_path: Optional[str] = None
    if results:
        best = max(results, key=lambda t: t[2])  # (start, end, score)
        s, e, sc = float(best[0]), float(best[1]), float(best[2])

        # Build output path
        #out_dir = CLIPS_DIR / _safe(query)
        out_dir = CLIPS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"best_score{sc:.3f}_{s:.2f}-{e:.2f}.mp4"

        try:
            saved_clip_path = _ffmpeg_cut(path, s, e, str(out_path))
            # Helpful response headers
            response.headers["X-Clip-Found"] = "1"
            response.headers["X-Clip-Start"] = f"{s:.2f}"
            response.headers["X-Clip-End"]   = f"{e:.2f}"
            response.headers["X-Clip-Score"] = f"{sc:.6f}"
            response.headers["X-Clip-Path"]  = saved_clip_path
        except Exception:
            # If cutting fails, we still return timestamps
            response.headers["X-Clip-Found"] = "0"
            saved_clip_path = None
    else:
        response.headers["X-Clip-Found"] = "0"

    # DELETE the uploaded file after clips are generated
    try:
        os.remove(path)
    except OSError:
        pass

    return PredictResponse(
        query=query,
        topk=topk,
        segments=[Segment(start=s, end=e, score=sc) for (s, e, sc) in results],
        saved_clip=saved_clip_path,
    
    
    )

#@app.post("/reload", summary="Reload model with a new config/checkpoint")
#def reload_model(body: ReloadBody):
    try:
        _load_model(body.config_path, body.checkpoint)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok", "device": str(_device)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference_api:app", host="0.0.0.0", port=8080, reload=True)

# Type this command on terminal:  uvicorn inference_api:app --host 0.0.0.0 --port 8000 --reload
