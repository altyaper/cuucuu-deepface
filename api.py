import io
import tempfile

import requests
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from detect_faces import detect_faces

app = FastAPI()


class DetectRequest(BaseModel):
    image_url: str


class FaceResult(BaseModel):
    confidence: float
    box: list[int]


class DetectResponse(BaseModel):
    face_count: int
    faces: list[FaceResult]


def _detect_from_bytes(image_bytes: bytes) -> DetectResponse:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        tmp.write(image_bytes)
        tmp.flush()
        raw_faces = detect_faces(tmp.name)

    faces = [
        FaceResult(confidence=round(f["confidence"], 4), box=list(f["box"]))
        for f in raw_faces
    ]
    return DetectResponse(face_count=len(faces), faces=faces)


@app.post("/detect", response_model=DetectResponse)
def detect_from_url(req: DetectRequest):
    print("Request for framing a face")
    resp = requests.get(req.image_url, timeout=15)
    resp.raise_for_status()
    return _detect_from_bytes(resp.content)


@app.post("/detect/upload", response_model=DetectResponse)
async def detect_from_upload(file: UploadFile = File(...)):
    image_bytes = await file.read()
    return _detect_from_bytes(image_bytes)


@app.get("/health")
def health():
    return {"status": "ok"}
