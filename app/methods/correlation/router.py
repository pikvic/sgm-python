from typing import Optional
from starlette import status
from fastapi import APIRouter
from fastapi import Request

from app.core.schema import TaskPostResult, MethodInfo
from app.core.queue import create_task
from .schema import CorrmatrixTaskParams, ScatterplotTaskParams
from .tasks import run_corrmatrix, run_scatterplot
from app.methods.methods import METHODS

router = APIRouter()

@router.post(
    "/corrmatrix",
    status_code=status.HTTP_201_CREATED,
    response_model=TaskPostResult
)
def corrmatrix_post(params: CorrmatrixTaskParams, request: Request):
    res = create_task(run_corrmatrix, params.dict())
    res['url'] = request.url_for('get_result', job_id=res['job_id'])
    return TaskPostResult(**res)

@router.get(
    "/corrmatrix"
)
def corrmatrix_get(request: Request):
    res = METHODS['scatterplot']
    res.image = str(request.base_url) + res.image
    return METHODS['corrmatrix']

@router.post(
    "/scatterplot",
    status_code=status.HTTP_201_CREATED,
    response_model=TaskPostResult
)
def scatterplot_post(params: ScatterplotTaskParams, request: Request):
    res = create_task(run_scatterplot, params.dict())
    res['url'] = request.url_for('get_result', job_id=res['job_id'])
    return TaskPostResult(**res)

@router.get(
    "/scatterplot"
)
def scatterplot_get():
    return METHODS['scatterplot']

