from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import numpy as np
from app.settings import settings

client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
COLLECTION = settings.QDRANT_COLLECTION

def init_db(vector_size=512):  # ✅ Add parameter
    if COLLECTION not in [col.name for col in client.get_collections().collections]:
        client.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )


def add_vector(id: int, vector: np.ndarray, payload: dict):
    client.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(id=id, vector=vector.tolist(), payload=payload)]
    )

def search_vector(vector: np.ndarray, top_k: int = 5):
    return client.search(
        collection_name=COLLECTION,
        query_vector=vector.tolist(),
        limit=top_k,
        with_payload=True,
        with_vectors=True     # ensures h.vector is returned
    )
