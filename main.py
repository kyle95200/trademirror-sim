# main.py
import os
import io
import uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# main.py
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "Backend running successfully"}

@app.post("/upload")
async def upload_file(file: UploadFile = None):
    content = await file.read()
    return {"filename": file.filename, "size": len(content)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000)

    # Hugging Face model endpoint
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    headers = {"Authorization": f"Bearer YOUR_HF_TOKEN"}

    response = requests.post(API_URL, headers=headers, json={"inputs": inputs})

    return response.json()


# Render needs this line to know which port to use
# It must be included in all Render FastAPI apps
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

# ====== ENV VARIABLES ======
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "charts")
EMBED_DIM = int(os.getenv("EMBED_DIM", "512"))

if not (QDRANT_URL and QDRANT_API_KEY):
    raise RuntimeError("Set QDRANT_URL and QDRANT_API_KEY as environment variables.")

# ====== INIT QDRANT ======
qclient = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ensure collection exists
try:
    qclient.get_collection(collection_name=COLLECTION)
except Exception:
    qclient.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

# ====== LOAD LOCAL CLIP MODEL ======
MODEL_NAME = "openai/clip-vit-base-patch32"

print(f"Loading CLIP model ({MODEL_NAME}) ... this may take a minute on first startup.")
clip_model = CLIPModel.from_pretrained(MODEL_NAME)
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
clip_model.eval()

def hf_image_to_vector(image_bytes: bytes):
    """Extract feature vector from an image using local CLIP model."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    vector = features.squeeze().cpu().numpy().tolist()
    return vector

# ====== ENDPOINTS ======
@app.post("/upsert/")
async def upsert_chart(
    file: UploadFile = File(...),
    ticker: str = Query(None),
    timeframe: str = Query(None),
    user_id: str = Query(None)
):
    """Store image embedding + metadata in Qdrant."""
    b = await file.read()
    vector = hf_image_to_vector(b)

    point_id = str(uuid.uuid4())
    payload = {"filename": file.filename, "ticker": ticker, "timeframe": timeframe, "user_id": user_id}
    point = PointStruct(id=point_id, vector=vector, payload=payload)

    qclient.upsert(collection_name=COLLECTION, points=[point])
    return {"status": "ok", "id": point_id, "payload": payload}

@app.post("/query/")
async def query_chart(file: UploadFile = File(...), top_k: int = 5):
    """Search for similar images in Qdrant."""
    b = await file.read()
    qvec = hf_image_to_vector(b)
    hits = qclient.search(collection_name=COLLECTION, query_vector=qvec, limit=top_k)
    results = [{"id": h.id, "score": h.score, "payload": h.payload} for h in hits]
    return {"matches": results}

@app.get("/")
def root():
    return {"message": "TradeMirror Similarity Engine running with local CLIP model."}
