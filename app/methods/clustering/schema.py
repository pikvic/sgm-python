from typing import Optional
from pydantic import BaseModel, Field, HttpUrl


class KMeansParams(BaseModel):
    url: HttpUrl
    nclusters: int = 6
    randomstate: Optional[int] = 0
    exclude: Optional[str] = Field(None, regex=r'^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$')
    addresultcolumns: Optional[bool] = False
    showstats: Optional[bool] = False
    normalize: Optional[bool] = False
    showgraph: Optional[bool] = False


class HCAParams(BaseModel):
    url: HttpUrl
    exclude: Optional[str] = Field(None, regex=r'^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$')
    normalize: Optional[bool] = False
