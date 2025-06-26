# Video Frame Vector Search Engine

This project implements a FastAPI-based system for uploading videos, extracting frames, generating feature vectors, storing them in a vector database (Qdrant), and allowing similarity-based search of frames.

## ✨ Features

- Upload a video file
- Automatically extract frames every N seconds
- Generate feature vectors from frames using color histogram
- Store vectors in Qdrant with associated metadata
- Search for similar frames using vector-based similarity
- Serve frame images as static files

---

## 📂 Project Structure

```
fastapi/
├── app/
│   ├── main.py           # FastAPI app & routes
│   ├── db.py             # Qdrant client setup & vector ops
│   ├── processing.py     # Frame extraction & vector computation
│   ├── schemas.py        # Request/response models
│   ├── settings.py       # Configuration settings
├── data/
│   ├── videos/           # Uploaded video files
│   └── frames/           # Extracted video frames
├── search_test.py        # Client to test vector search API
├── requirements.txt      # All required Python packages
└── README.md             # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone Repository

```bash
git clone https://github.com/ErrorSol/video-frame-search.git
cd fastapi
```

### 2. Set up Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start Qdrant (Docker Required)

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 4. Start the FastAPI Server

```bash
uvicorn app.main:app --reload
```

Visit: [http://localhost:8000/docs](http://localhost:8000/docs) to access Swagger UI

### 5. Test Vector Search

Make sure `data/frames/frame_0.jpg` exists, then:

```bash
python search_test.py
```

---

## 🔍 API Endpoints

### `GET /`

Basic health check.

### `POST /upload`

Upload a video file and extract frames.

- **file**: Video file (UploadFile)
- **interval**: Interval in seconds between extracted frames

### `POST /vector/search`

Search for similar frames given a feature vector.

- **vector**: List[float]
- **top\_k**: Number of similar frames to return

### `GET /frames/<filename>`

Access extracted frames as static images.

---

## 📄 Requirements

Generated via:

```bash
pip freeze > requirements.txt
```

Example:

```txt
fastapi
uvicorn
qdrant-client
numpy
opencv-python
requests
```

---

## 🌍 Credits

Developed as part of internship/assignment project to demonstrate skills in:

- Python APIs
- Vector databases
- Computer vision
- Async processing

---

## 🚫 Disclaimer

Only use the tool on publicly shareable videos. Avoid using sensitive or private data.

