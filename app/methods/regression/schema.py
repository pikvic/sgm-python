from typing import Optional
from pydantic import BaseModel, Field, HttpUrl


class LinearParams(BaseModel):
    url: HttpUrl
    xcolumn: int = Field(..., gt=0)
    ycolumn: int = Field(..., gt=0)
