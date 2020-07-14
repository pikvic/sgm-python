from typing import List
from pydantic import BaseModel

class TaskPostResult(BaseModel):
    job_id: str
    url: str

class TaskResult(BaseModel):
    ready: bool = False
    results: List[str] = None