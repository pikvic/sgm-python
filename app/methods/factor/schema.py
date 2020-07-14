from typing import Optional
from pydantic import BaseModel, Field, HttpUrl


class PCAParams(BaseModel):
    url: HttpUrl
    exclude: Optional[str] = Field(None, regex=r'^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$')
    normalize: Optional[bool] = False
    showgraph: Optional[bool] = False