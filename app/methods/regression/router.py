from typing import Optional
from starlette import status
from fastapi import APIRouter
from fastapi import Request

from app.core.schema import TaskPostResult, MethodInfo
from app.core.queue import create_task
from .schema import LinearRegressionTaskParams, PolynomialRegressionTaskParams
from .tasks import run_linearregression, run_polynomialregression
from app.methods.methods import METHODS

router = APIRouter()

@router.post(
    "/linearregression",
    status_code=status.HTTP_201_CREATED,
    response_model=TaskPostResult
)
def linearregression_post(params: LinearRegressionTaskParams, request: Request):
    res = create_task(run_linearregression, params.dict())
    res['url'] = request.url_for('get_result', job_id=res['job_id'])
    return TaskPostResult(**res)

@router.get(
    "/linearregression"
)
def linearregression_get(request: Request):
    res = METHODS['linearregression']
    res.image = str(request.base_url) + res.image
    return METHODS['linearregression']

@router.post(
    "/polynomialregression",
    status_code=status.HTTP_201_CREATED,
    response_model=TaskPostResult
)
def polynomialregression_post(params: PolynomialRegressionTaskParams, request: Request):
    res = create_task(run_polynomialregression, params.dict())
    res['url'] = request.url_for('get_result', job_id=res['job_id'])
    return TaskPostResult(**res)

@router.get(
    "/polynomialregression"
)
def polynomialregression_get():
    return METHODS['polynomialregression']
