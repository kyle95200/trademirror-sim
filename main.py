from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can tighten this later for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "Backend running successfully"}

@app.post("/upload")
async def upload_file(file: UploadFile = None):
    if not file:
        return {"error": "No file uploaded"}
    content = await file.read()
    return {"filename": file.filename, "size": len(content)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
