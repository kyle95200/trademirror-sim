# main.py
import os
import io
import uuid
import requests
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from qdrant_client import QdrantClient

# Initialize the Qdrant client with your credentials
qdrant_client = QdrantClient(
    url="https://5030f791-3e14-4605-a543-fb42d57e0ba3.eu-west-2-0.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.qa5Q9JFbj-Gs3N5BdGUFEv6yPeCoUXrYelZ1TrMnS7Y",
)

from qdrant_client.http.models import VectorParams, Distance, PointStruct
from PIL import Image

app = FastAPI(title="TradeMirror Similarity Engine")

# ENV: set these in Render dashboard (or .env for local testing)
HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face token
QDRANT_URL = os.getenv("QDRANT_URL")  # e.g. "https://xxxxxxx-xxx.qdrant.cloud"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "charts")
EMBED_DIM = int(os.getenv("EMBED_DIM", "512"))  # adjust if you use a different model

if not (HF_TOKEN and QDRANT_URL and QDRANT_API_KEY):
    raise RuntimeError("Set HF_TOKEN, QDRANT_URL and QDRANT_API_KEY as env vars")

# init Qdrant client
qclient = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ensure collection exists (recreate if not)
try:
    qclient.get_collection(collection_name=COLLECTION)
except Exception:
    qclient.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),)
HF_MODEL = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"  # example; pick a model that supports feature-extraction

def hf_image_to_vector(image_bytes: bytes):
    """
    Post image bytes to Hugging Face Inference API for feature-extraction.
    Note: model must expose feature-extraction for images.
    """
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    resp = requests.post(url, headers=headers, data=image_bytes)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"HuggingFace error: {resp.status_code} {resp.text[:200]}")
    data = resp.json()
    # Many HF 'feature-extraction' responses are nested lists; flatten if needed
    # Example: [[0.1, 0.2, ...]] or a single list.
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        vec = np.array(data).mean(axis=0)  # if returned as per-patch features, average
    else:
        vec = np.array(data)
    # ensure correct shape and type
    vec = vec.astype(float)
    if vec.ndim > 1:
        vec = vec.reshape(-1)
    return vec.tolist()

@app.post("/upsert/")
async def upsert_chart(file: UploadFile = File(...), ticker: str = Query(None), timeframe: str = Query(None), user_id: str = Query(None)):
    # read image bytes
    b = await file.read()
    # optional: validate image
    try:
        Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    vector = hf_image_to_vector(b)
    point_id = str(uuid.uuid4())
    payload = {"filename": file.filename, "ticker": ticker, "timeframe": timeframe, "user_id": user_id}
    point = PointStruct(id=point_id, vector=vector.tolist() if isinstance(vector, np.ndarray) else vector, payload=payload)
    qclient.upsert(collection_name=COLLECTION, points=[point])
    return {"status": "ok", "id": point_id, "payload": payload}

@app.post("/query/")
async def query_chart(file: UploadFile = File(...), top_k: int = 5):
    b = await file.read()
    try:
        Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    qvec = hf_image_to_vector(b)
    hits = qclient.search(collection_name=COLLECTION, query_vector=qvec.tolist(), limit=top_k)
    # simplify the response for frontend
    results = [{"id": h.id, "score": h.score, "payload": h.payload} for h in hits]
    return {"matches": results}
@app.get("/")
async def root():
    return {"status": "ok", "message": "TradeMirror backend is running"}
