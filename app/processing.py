import cv2
import numpy as np
from pathlib import Path

def extract_frames(video_path: Path, out_dir: Path, every_n_sec: int = 1):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = int(fps * every_n_sec)

    frame_id, saved = 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id % interval == 0:
            p = out_dir / f"frame_{int(frame_id // fps)}.jpg"

            cv2.imwrite(str(p), frame)
            saved.append(p)
        frame_id += 1
    cap.release()
    return saved

def compute_feature_vector(img_path: Path) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to read {img_path}")
    hist = cv2.calcHist([img], [0,1,2], None, [8,8,8], [0,256]*3)
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype("float32")
