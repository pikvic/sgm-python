import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import app.core.config as config
from app.core.tasks import validate_input_and_get_dataframe, get_or_create_dir


def run_pca(params):
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
        if params['exclude']:
            pattern = r'^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$'
            if not re.search(pattern, params['exclude']):
                return {'success': False, 'error': 'Wrong exclude columns pattern!'}

            res = re.findall(r'\d+(?:-\d+)*', params['exclude'])
            columns = set()
            for r in res:
                if '-' in r:
                    left, right = r.split('-')
                    if left < right:
                        columns = columns | set(range(int(left) - 1, int(right)))
                else:
                    columns.add(int(r) - 1)
            columns = list({i for i in range(len(df.columns))} - columns)
            data = df.iloc[:, columns]
        else:
            data = df.iloc[:, :]        
    except:
        return {'success': False, 'error': 'Error while excluding columns!'}    

    columns = data.columns
    if params['normalize']:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    try:
        file_path = root / 'components.csv'
        pca = PCA()
        pca.fit(data)
        components = pd.DataFrame(pca.components_, columns=columns)
        components.index.name = 'Component'
        file_path = root / 'components.csv'
        components.to_csv(file_path)
        results.append(str(file_path))

        file_path = root / 'variance.csv'
        pd.DataFrame({'Explained Variance': pca.explained_variance_,
                'Explained Variance Ratio': pca.explained_variance_ratio_}).to_csv(file_path, index=False)
        results.append(str(file_path))
    except:
        return {'success': False, 'error': 'Error while saving result!'}    

    params['showgraph'] = True
    if params['showgraph']:
        try:
            fig, ax = plt.subplots()
            ax.bar(list(range(1, pca.n_components_ + 1)), pca.explained_variance_ratio_)
            ax.set_title('Principal Component Analysis')
            ax.set_xlabel('Number of components')
            ax.set_ylabel('Explained Variance Ratio')
            file_path = root / 'pca_figure_1.png'
            fig.savefig(file_path, bbox_inches = 'tight')
            results.append(str(file_path))
            fig, ax = plt.subplots()
            ax.plot(list(range(1, pca.n_components_ + 1)), np.cumsum(pca.explained_variance_ratio_))
            ax.set_title('Principal Component Analysis')
            ax.set_xlabel('Number of components')
            ax.set_ylabel('Cumulative Explained Variance Ratio')
            file_path = root / 'pca_figure_2.png'
            fig.savefig(file_path, bbox_inches = 'tight')
            results.append(str(file_path))
        except:
            return {'success': False, 'error': 'Error while showing graph!'}
    return {'ready': True, 'results': results}