from typing import Optional
from starlette import status
from fastapi import APIRouter
from app.core.schema import TaskPostResult
from app.core.queue import create_task
from .schema import StatsTaskParams
from .tasks import run_stats

router = APIRouter()

@router.post(
    "/stats",
    status_code=status.HTTP_201_CREATED,
    response_model=TaskPostResult,
    summary="Описатиельная статистика",
    description="Вычисление описательной статистики по входным данным"
)
def stats(params: StatsTaskParams):
    res = create_task(run_stats, params.dict())
    return TaskPostResult(**res)