import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import app.core.config as config
from app.core.tasks import validate_input_and_get_dataframe, get_or_create_dir

def run_stats(params):
    # common
    results = []
    job_id = params['job_id']
    res = validate_input_and_get_dataframe(params['url'], job_id)
    if not res['success']:
        return res
    df = res['dataframe']
    root = get_or_create_dir(config.DOWNLOAD_DIR, job_id)

    # specific
    try:
        column = params['column'] - 1
        data = df.iloc[:, column]
    except:
        return {'success': False, 'error': 'Wrong column parameter!'}
    stats = data.describe()
    stats.index.name = "Stats"
    
    try:
        output = 'result.csv'
        file_path = root / output
        if params['transpose']:
            stats.T.to_csv(file_path)
        else:
            stats.to_csv(file_path)
        results.append(str(file_path))
    except:
        return {'success': False, 'error': 'Error while saving result!'}

    if params['showgraph']:
        try:
            sns.set_style("whitegrid")
            sns.set_context("talk")
            fig, ax = plt.subplots(2, 1, figsize=(6, 10))
            sns.distplot(data, ax=ax[0])
            sns.boxplot(data, ax=ax[1])
            figname = f'Column_{column + 1}.png'
            file_path = root / figname
            fig.savefig(file_path, bbox_inches = 'tight')
            results.append(str(file_path))
            plt.close(fig)
        except:
            return {'success': False, 'error': 'Error while showing graph!'}
    return {'ready': True, 'results': results}