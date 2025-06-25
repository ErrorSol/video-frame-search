from pathlib import Path
import shutil, uuid, asyncio
import numpy as np

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles

from app.processing import extract_frames, compute_feature_vector
from app.db import init_db, add_vector, search_vector
from app.schemas import SearchRequest, SearchHit
from app.settings import settings

UPLOAD_DIR = Path("data/videos")
FRAME_DIR  = Path("data/frames")

app = FastAPI(title="Video-Frame Vector Search")

app.mount("/frames", StaticFiles(directory="data/frames"), name="frames")

@app.on_event("startup")
def on_startup() -> None:
    init_db(vector_size=512)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)

def _store_frames_and_vectors(video_path: Path, every_n_sec: int = 1) -> int:
    frames = extract_frames(video_path, FRAME_DIR, every_n_sec)
    for frame_path in frames:
        vec = compute_feature_vector(frame_path)
        add_vector(
            id=int(uuid.uuid4().int >> 64),
            vector=vec,
            payload={"frame_path": str(frame_path)}
        )
    return len(frames)

@app.get("/", tags=["Health"])
def root():
    return {"status": "OK", "docs": "/docs"}

@app.post("/upload", tags=["Ingest"])
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    interval: int = 1
):
    dst = UPLOAD_DIR / file.filename
    with open(dst, "wb") as fout:
        shutil.copyfileobj(file.file, fout)

    loop = asyncio.get_running_loop()
    frame_count = await loop.run_in_executor(
        None, _store_frames_and_vectors, dst, interval
    )

    return {
        "message": "Video uploaded",
        "frames_extracted": frame_count,
        "interval_sec": interval
    }

@app.post("/vector/search", response_model=list[SearchHit])
async def search_similar(req: SearchRequest):
    hits = search_vector(np.array(req.vector, dtype="float32"), top_k=req.top_k)
    return [
        SearchHit(
            score=h.score,
            frame_path=h.payload.get("frame_path"),
            vector=h.vector,
        )
        for h in hits
    ]
