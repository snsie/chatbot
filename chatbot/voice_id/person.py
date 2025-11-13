from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import List

class EmbeddingData(BaseModel):
    vec: List[float]
    model: str

class Person(BaseModel):
    name: str
    current_embedding: EmbeddingData
    file_path: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))