from typing import Optional
from starlette import status
from fastapi import APIRouter

from app.core.schema import TaskPostResult
from app.core.queue import create_task

from .schema import KMeansParams, HCAParams
from .tasks import run_kmeans, run_hca

router = APIRouter()

@router.post(
    "/kmeans",
    status_code=status.HTTP_201_CREATED,
    response_model=TaskPostResult,
    summary="Метод К-средних",
    description="Кластеризация методом К-средних"
)
def kmeans(params: KMeansParams):
    res = create_task(run_kmeans, params.dict())
    return TaskPostResult(**res)

@router.post(
    "/hca",
    status_code=status.HTTP_201_CREATED,
    response_model=TaskPostResult,
    summary="Метод иерархического кластерного анализа",
    description="Кластеризация методом иерархического кластерного анализа"
)
def hca(params: HCAParams):
    res = create_task(run_hca, params.dict())
    return TaskPostResult(**res)

