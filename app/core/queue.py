from redis import from_url
from rq import Queue
from rq.job import Job
from uuid import uuid4
from datetime import timedelta
import app.core.config as config
from app.core.tasks import clear_files_for_job

red = from_url(config.REDIS_URL)
queue = Queue(connection=red, default_timeout=3600)


def generate_id():
    return str(uuid4())

def create_task(func, params):
    job_id = generate_id()
    params['job_id'] = job_id
    job = queue.enqueue_call(func=func, args=(params,), result_ttl=config.RESULT_TTL, job_id=job_id)
    job_clear_files = queue.enqueue_in(timedelta(seconds=config.RESULT_TTL), func=clear_files_for_job, args=(job_id,), result_ttl=config.RESULT_TTL)
    print(job_clear_files)
    res = {"job_id": job_id, "url": f"/results/{job_id}"}
    return res

def get_redis():
    return red

def get_queue():
    return queue