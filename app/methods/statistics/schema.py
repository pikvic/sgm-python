from typing import Optional
from pydantic import BaseModel, Field, HttpUrl


class StatsTaskParams(BaseModel):
    url: HttpUrl
    column: int = Field(..., gt=0)
    transpose: Optional[bool] = False
    showgraph: Optional[bool] = False