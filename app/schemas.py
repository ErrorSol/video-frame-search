from pydantic import BaseModel, Field
from typing import List, Optional

class SearchRequest(BaseModel):
    vector: List[float] = Field(..., min_items=512, max_items=512)
    top_k: int = Field(5, ge=1, le=50)

class SearchHit(BaseModel):
    score: float
    frame_path: Optional[str] = None
    vector: List[float]
