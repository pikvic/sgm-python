from pydantic import BaseModel, Field


class TestTaskParams(BaseModel):
    seconds: int = Field(..., title='Seconds to sleep', gt=0, lt=20)    