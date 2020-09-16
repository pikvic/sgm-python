from typing import Optional
from starlette import status
from fastapi import APIRouter
from fastapi import Request

from app.core.schema import TaskPostResult, MethodInfo
from app.core.queue import create_task
from .schema import FactorAnalysisTaskParams, FactorScreePlotTaskParams, PcaChooseTaskParams, PcaTaskParams
from .tasks import run_factoranalysis, run_factorscreeplot, run_pca, run_pcachoose
from app.methods.methods import METHODS

router = APIRouter()

@router.post(
    "/factoranalysis",
    status_code=status.HTTP_201_CREATED,
    response_model=TaskPostResult
)
def factoranalysis_post(params: FactorAnalysisTaskParams, request: Request):
    res = create_task(run_factoranalysis, params.dict())
    res['url'] = request.url_for('get_result', job_id=res['job_id'])
    return TaskPostResult(**res)

@router.get(
    "/factoranalysis"
)
def factoranalysis_get(request: Request):
    res = METHODS['factoranalysis']
    res.image = str(request.base_url) + res.image
    return METHODS['factoranalysis']

@router.post(
    "/factorscreeplot",
    status_code=status.HTTP_201_CREATED,
    response_model=TaskPostResult
)
def factorscreeplot_post(params: FactorScreePlotTaskParams, request: Request):
    res = create_task(run_factorscreeplot, params.dict())
    res['url'] = request.url_for('get_result', job_id=res['job_id'])
    return TaskPostResult(**res)

@router.get(
    "/factorscreeplot"
)
def scatterplot_get():
    return METHODS['factorscreeplot']

@router.post(
    "/pca",
    status_code=status.HTTP_201_CREATED,
    response_model=TaskPostResult
)
def pca_post(params: PcaTaskParams, request: Request):
    res = create_task(run_pca, params.dict())
    res['url'] = request.url_for('get_result', job_id=res['job_id'])
    return TaskPostResult(**res)

@router.get(
    "/pca"
)
def pca_get(request: Request):
    res = METHODS['pca']
    res.image = str(request.base_url) + res.image
    return METHODS['pca']

@router.post(
    "/pcachoose",
    status_code=status.HTTP_201_CREATED,
    response_model=TaskPostResult
)
def pcachoose_post(params: PcaChooseTaskParams, request: Request):
    res = create_task(run_pcachoose, params.dict())
    res['url'] = request.url_for('get_result', job_id=res['job_id'])
    return TaskPostResult(**res)

@router.get(
    "/pcachoose"
)
def pcachoose_get(request: Request):
    res = METHODS['pcachoose']
    res.image = str(request.base_url) + res.image
    return METHODS['pcachoose']