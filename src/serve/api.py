"""FastAPI visual search service: POST an image, get top-K similar products.

Loads the trained embedder + FAISS gallery index once at startup. Config comes
from env vars (set by docker-compose): CKPT, INDEX_DIR, DATA_ROOT, DEVICE.
"""

from __future__ import annotations

import os

# torch/faiss/opencv OpenMP clash workaround (see src/index/build.py); harmless on Linux.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import io
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from src.data.augmentations import build_eval_transform
from src.models.embedder import build_embedder, resolve_device

faiss.omp_set_num_threads(1)

CKPT = Path(os.getenv("CKPT", "models/best.pt"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", "results/index"))
DATA_ROOT = Path(os.getenv("DATA_ROOT", "data/Stanford_Online_Products")).resolve()
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))

app = FastAPI(title="Semblance — visual product search")
_state: dict = {}


@app.on_event("startup")
def _load() -> None:
    device = resolve_device(os.getenv("DEVICE", "cpu"))
    model = build_embedder(EMBEDDING_DIM).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()
    _state.update(
        device=device,
        model=model,
        transform=build_eval_transform(),
        index=faiss.read_index(str(INDEX_DIR / "gallery.faiss")),
        meta=pd.read_csv(INDEX_DIR / "gallery_meta.csv"),
    )
    print(f"loaded model + index ({_state['index'].ntotal:,} products) on {device.type}")


@torch.no_grad()
def _embed(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"))
    x = _state["transform"](image=arr)["image"].unsqueeze(0).to(_state["device"])
    return _state["model"](x).cpu().numpy().astype("float32")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "gallery_size": _state["index"].ntotal}


@app.post("/search")
async def search(file: UploadFile = File(...), k: int = Query(12, ge=1, le=50)) -> dict:
    """Embed the uploaded image and return the k nearest gallery products."""
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "could not read image")
    scores, idx = _state["index"].search(_embed(img), k)
    meta = _state["meta"]
    results = []
    for rank, (score, i) in enumerate(zip(scores[0], idx[0]), start=1):
        row = meta.iloc[int(i)]
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "path": row["path"],
                "category": row["category"],
                "class_id": int(row["class_id"]),
            }
        )
    return {"results": results}


@app.get("/image")
def image(path: str) -> FileResponse:
    """Serve a gallery image, sandboxed to DATA_ROOT to block path traversal."""
    full = (DATA_ROOT / path).resolve()
    if DATA_ROOT not in full.parents or not full.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(full)
