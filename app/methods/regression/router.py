from typing import Optional
from starlette import status
from fastapi import APIRouter

from app.core.schema import TaskPostResult
from app.core.queue import create_task

from .schema import LinearParams
from .tasks import run_linear

router = APIRouter()

@router.post(
    "/linear",
    status_code=status.HTTP_201_CREATED,
    response_model=TaskPostResult,
    summary="Линейная регрессия",
    description="Построение линейной регрессии"
)
def linear(params: LinearParams):
    res = create_task(run_linear, params.dict())
    return TaskPostResult(**res)