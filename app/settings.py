from pydantic_settings import BaseSettings  # pydantic-settings v2
from pydantic import Field

class Settings(BaseSettings):
    QDRANT_HOST: str = Field("localhost", env="QDRANT_HOST")
    QDRANT_PORT: int = Field(6333, env="QDRANT_PORT")
    QDRANT_COLLECTION: str = Field("video_frames", env="QDRANT_COLLECTION")

    UPLOAD_DIR: str = "data/videos"
    FRAME_DIR: str = "data/frames"
    VECTOR_SIZE: int = 512  # 8×8×8 RGB histogram

    class Config:
        env_file = ".env"

settings = Settings()
