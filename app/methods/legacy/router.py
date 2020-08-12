from typing import Optional
from starlette import status
from fastapi import APIRouter

from app.core.schema import TaskPostResult
from app.core.queue import create_task

from .schema import LinearParams, HCAParams, PCAParams, KMeansParams, StatsTaskParams
from .tasks import run_linear, run_hca, run_kmeans, run_pca, run_stats

router = APIRouter()


@router.post(
    "/clustering/kmeans",
    status_code=status.HTTP_201_CREATED,
    response_model=TaskPostResult,
    summary="Метод К-средних",
    description="Кластеризация методом К-средних",
    tags=["clustering"]
)
def kmeans(params: KMeansParams):
    res = create_task(run_kmeans, params.dict())
    return TaskPostResult(**res)

@router.post(
    "/clustering/hca",
    status_code=status.HTTP_201_CREATED,
    response_model=TaskPostResult,
    summary="Метод иерархического кластерного анализа",
    description="Кластеризация методом иерархического кластерного анализа",
    tags=["clustering"]
)
def hca(params: HCAParams):
    res = create_task(run_hca, params.dict())
    return TaskPostResult(**res)


@router.post(
    "/regression/linear",
    status_code=status.HTTP_201_CREATED,
    response_model=TaskPostResult,
    summary="Линейная регрессия",
    description="Построение линейной регрессии",
    tags=["regression"]
)
def linear(params: LinearParams):
    res = create_task(run_linear, params.dict())
    return TaskPostResult(**res)

@router.post(
    "/statistics/stats",
    status_code=status.HTTP_201_CREATED,
    response_model=TaskPostResult,
    summary="Описатиельная статистика",
    description="Вычисление описательной статистики по входным данным",
    tags=["statistics"]
)
def stats(params: StatsTaskParams):
    res = create_task(run_stats, params.dict())
    return TaskPostResult(**res)

@router.post(
    "/factor/pca",
    status_code=status.HTTP_201_CREATED,
    response_model=TaskPostResult,
    summary="Метод главных компонент",
    description="Выделение главных компонент",
    tags=["factor"]
)
def pca(params: PCAParams):
    res = create_task(run_pca, params.dict())
    return TaskPostResult(**res)