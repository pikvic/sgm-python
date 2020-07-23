import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import app.core.config as config
from app.core.tasks import validate_input_and_get_dataframe, get_or_create_dir


def run_linear(params):
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
        data = df.iloc[:, :]
        x_column_id = params['xcolumn'] - 1
        y_column_id = params['ycolumn'] - 1
        x_column_id = x_column_id if x_column_id < len(data.columns) else 0
        y_column_id = y_column_id if y_column_id < len(data.columns) else 1

        x = data.iloc[:, x_column_id].values.reshape(-1, 1)
        y = data.iloc[:, y_column_id].values.reshape(-1, 1)

        lr = LinearRegression()
        lr.fit(x, y)
        yhat = lr.predict(x)

        res = {
            'Parameter': ['Coefficient', 'Intercept', 'Mean Squared Error'],
            'Value': [lr.coef_[0][0], lr.intercept_[0], mean_squared_error(y, yhat)]
            }
        file_path = root / 'result.csv'
        pd.DataFrame(res).to_csv(file_path, index=False)
        results.append(str(file_path))
    except Exception as e:
        return {'success': False, 'error': f'Error while saving result! Exception: {e}'}

    try:
        fig, ax = plt.subplots()
        sns.regplot(x, y, yhat, ax=ax)
        ax.set(title=f'y = {lr.coef_[0][0]:.4f}x + {lr.intercept_[0]:.4f}')
        ax.set(xlabel=data.columns[x_column_id], ylabel=data.columns[y_column_id])
        figname = f'Regression_Column{x_column_id + 1}_Column{y_column_id + 1}.png'
        file_path = root / figname
        fig.savefig(file_path, bbox_inches = 'tight')
        results.append(str(file_path))
    except:
        return {'success': False, 'error': 'Error while showing graph!'}

    return {'ready': True, 'results': results}