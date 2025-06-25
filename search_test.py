import requests
import json
import numpy as np

# Use a sample random 512-d vector just for testing
vec = np.random.rand(512).astype("float32").tolist()

resp = requests.post(
    "http://localhost:8000/vector/search",
    json={"vector": vec, "top_k": 5},
)

print("Search results:")
print(json.dumps(resp.json(), indent=2))
