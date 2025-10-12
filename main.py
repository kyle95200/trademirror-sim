# main.py
import os
import io
import uuid
import json
import requests
import numpy as np
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from PIL import Image

# Optional heavy deps for local model
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").lower() in ("1", "true", "yes")
HF_TOKEN = os.getenv("HF_TOKEN")            # required if using remote HF
HF_MODEL = os.getenv("HF_MODEL", "clip-vit-base-patch32")  # used for remote call if not using full endpoint
HF_ENDPOINT = os.getenv("HF_ENDPOINT")      # optional explicit HF endpoint (preferred if you deployed an endpoint)
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "charts")
EMBED_DIM = int(os.getenv("EMBED_DIM", "512"))

# Validate minimal envs
if not (QDRANT_URL and QDRANT_API_KEY):
    raise RuntimeError("Set QDRANT_URL and QDRANT_API_KEY as environment variables.")

if (not USE_LOCAL_MODEL) and (not HF_TOKEN):
    raise RuntimeError("When USE_LOCAL_MODEL is false, set HF_TOKEN environment variable for Hugging Face remote inference.")

app = FastAPI(title="TradeMirror Similarity Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # relax for now; lock down in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== QDRANT INIT =====
qclient = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

try:
    qclient.get_collection(collection_name=COLLECTION)
except Exception:
    qclient.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

# ===== MODEL (local optional) =====
clip_model = None
clip_processor = None
use_remote_url = None

if USE_LOCAL_MODEL:
    # lazy import heavy libs inside try to give clearer errors
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
    except Exception as e:
        raise RuntimeError("To use local model you must include 'transformers' and 'torch' in requirements.") from e

    MODEL_NAME = os.getenv("LOCAL_CLIP_MODEL", "openai/clip-vit-base-patch32")
    print(f"[startup] Loading local CLIP model '{MODEL_NAME}' (this may take a minute)...")
    clip_model = CLIPModel.from_pretrained(MODEL_NAME)
    clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    clip_model.eval()
    # keep torch on CPU by default; consider GPU if available and configured in Render
    print("[startup] Local CLIP model loaded.")
else:
    # remote mode: determine which HF URL to call
    # prefer HF_ENDPOINT (user-deployed endpoint); otherwise use standard inference api with HF_MODEL
    if HF_ENDPOINT:
        use_remote_url = HF_ENDPOINT.rstrip("/")  # user-provided full URL
    else:
        use_remote_url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

    # sanity print
    print(f"[startup] Using Hugging Face remote inference URL: {use_remote_url}")

# ===== Helper functions =====
def flatten_vector_from_hf_response(data):
    """
    Helper to normalize HuggingFace inference responses to a flat vector list.
    Many HF image embedding endpoints return nested lists.
    """
    arr = np.array(data)
    if arr.ndim > 1:
        arr = arr.mean(axis=0)
    return arr.astype(float).reshape(-1).tolist()

def local_image_to_vector_sync(image_bytes: bytes):
    """Blocking local inference; run in threadpool via run_in_threadpool."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    vec = features.squeeze().cpu().numpy()
    return vec.astype(float).reshape(-1).tolist()

def remote_image_to_vector_sync(image_bytes: bytes):
    """
    Send binary image bytes to HF inference endpoint (content-type application/octet-stream).
    If the HF endpoint requires JSON, adapt — but most image inference endpoints accept raw binary.
    """
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    # If user configured an HF_ENDPOINT that expects JSON, they can set HF_ENDPOINT accordingly.
    resp = requests.post(use_remote_url, headers=headers, data=image_bytes, timeout=60)
    if resp.status_code != 200:
        # return useful diagnostic in logs and raise HTTPException
        msg = resp.text
        print(f"[hf error] status {resp.status_code} body: {msg}")
        raise HTTPException(status_code=502, detail=f"HuggingFace error: {resp.status_code} {msg[:500]}")
    data = resp.json()
    # Many HF inference image-to-vec endpoints return nested list(s) — normalize:
    return flatten_vector_from_hf_response(data)

async def image_to_vector(image_bytes: bytes):
    """Async wrapper that dispatches to local or remote inference and runs blocking ops in threadpool."""
    if USE_LOCAL_MODEL:
        # local heavy compute -> run in threadpool
        return await run_in_threadpool(local_image_to_vector_sync, image_bytes)
    else:
        # remote call (network I/O) -> run in threadpool to avoid blocking
        return await run_in_threadpool(remote_image_to_vector_sync, image_bytes)

# ===== Endpoints =====
@app.get("/")
def health():
    return {"status": "ok", "mode": "local" if USE_LOCAL_MODEL else "remote"}

@app.post("/upsert/")
async def upsert_chart(
    file: UploadFile = File(...),
    ticker: Optional[str] = Query(None),
    timeframe: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
):
    b = await file.read()
    # validate image
    try:
        Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # get vector
    try:
        vector = await image_to_vector(b)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[error] image->vector failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to extract image vector")

    # check vector dimension (optional)
    if len(vector) != EMBED_DIM:
        # if mismatch, allow but log
        print(f"[warning] vector length {len(vector)} != EMBED_DIM {EMBED_DIM}")

    point_id = str(uuid.uuid4())
    payload = {"filename": file.filename, "ticker": ticker, "timeframe": timeframe, "user_id": user_id}
    point = PointStruct(id=point_id, vector=vector, payload=payload)

    qclient.upsert(collection_name=COLLECTION, points=[point])
    return {"status": "ok", "id": point_id, "payload": payload}

@app.post("/query/")
async def query_chart(file: UploadFile = File(...), top_k: int = Query(5, gt=0, le=100)):
    b = await file.read()
    try:
        Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        qvec = await image_to_vector(b)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[error] query image->vector failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to extract image vector")

    hits = qclient.search(collection_name=COLLECTION, query_vector=qvec, limit=top_k)
    results = [{"id": h.id, "score": h.score, "payload": h.payload} for h in hits]
    return {"matches": results}

# ===== Run server when executed directly (Render will use uvicorn main:app) =====
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
