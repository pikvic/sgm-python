import app.core.config as config
from time import sleep

def run_test(params):
    sleep(params['seconds'])
    results = [f"Sleeped {params['seconds']} seconds."]
    return {'ready': True, 'results': results}

