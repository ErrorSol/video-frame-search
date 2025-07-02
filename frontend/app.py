import streamlit as st
import requests
import numpy as np
import os

BACKEND_URL = "http://localhost:8000"

st.title("🎬 Video Frame Search")

st.sidebar.header("Upload a video")
uploaded_video = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
interval = st.sidebar.slider("Frame Interval (seconds)", 1, 10, 1)

if uploaded_video:
    st.video(uploaded_video)
    if st.sidebar.button("Upload and Extract Frames"):
        st.spinner("Uploading and processing...")
        files = {"file": (uploaded_video.name, uploaded_video, uploaded_video.type)}
        resp = requests.post(f"{BACKEND_URL}/upload", files=files, data={"interval": interval})
        if resp.ok:
            res = resp.json()
            st.success(f"✅ {res['frames_extracted']} frames extracted.")
            st.subheader("Extracted Frames")
            for fname in res["frame_paths"]:
                st.image(f"{BACKEND_URL}/frames/{fname}", caption=fname, use_container_width=True)
        else:
            st.error("❌ Upload failed.")

st.markdown("---")
st.header("🔍 Search Frames")

if st.button("Search using random vector"):
    vec = np.random.rand(512).astype("float32").tolist()
    r = requests.post(f"{BACKEND_URL}/vector/search", json={"vector": vec, "top_k": 5})
    if r.ok:
        hits = r.json()
        for h in hits:
            st.image(f"{BACKEND_URL}/frames/{os.path.basename(h['frame_path'])}",
                     caption=f"Score: {h['score']:.4f}", use_container_width=True)
    else:
        st.error("Search failed.")
