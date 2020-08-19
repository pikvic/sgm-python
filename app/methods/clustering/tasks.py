import re
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import app.core.config as config
from app.core.tasks import validate_input_and_get_dataframe, get_or_create_dir, generate_filename, error, ready, validate_columns_params, parse_columns


def run_kmeans(params):
    # common
    results = []
    job_id = params['job_id']
    res = validate_input_and_get_dataframe(params['url'], job_id)
    if not res['success']:
        return res
    df = res['dataframe']
    root = get_or_create_dir(config.DOWNLOAD_DIR, job_id)

    # validate params
    res = validate_columns_params(params, len(df.columns))
    if not res['success']:
        return res

    # specific
    columns = parse_columns(params['columns'], len(df.columns))['data']

    try:
        data = df.iloc[:, columns]
        model = KMeans(n_clusters=params['nclusters'])
        model.fit(data)
        labels = model.labels_ + 1
        centers = model.cluster_centers_
        clusters_df = pd.DataFrame(data=labels, columns=['Номер кластера'])
        centers_df = pd.DataFrame(data=centers, columns=data.columns)
        output_df = data.copy()
        output_df['Номер кластера'] = labels
    except Exception as e:
        return error(f'Ошибка во время работы алгоритма k-средних : {e}')  
   
    # save output
    try:
        if params['file_format'] == 'CSV':
            file_path = generate_filename(root, 'summary', 'clusters.csv')
            clusters_df.to_csv(file_path, index=False)
            results.append(str(file_path))
            file_path = generate_filename(root, 'summary', 'cluster_centers.csv')
            centers_df.to_csv(file_path, index=False)
            results.append(str(file_path))
            file_path = generate_filename(root, 'summary', 'input_with_clusters.csv')
            output_df.to_csv(file_path, index=False)
            results.append(str(file_path))
        elif params['file_format'] == 'XLSX':
            file_path = generate_filename(root, 'summary', 'clusters.xlsx')
            clusters_df.to_excel(file_path, index=False)
            results.append(str(file_path))
            file_path = generate_filename(root, 'summary', 'cluster_centers.xlsx')
            centers_df.to_excel(file_path, index=False)
            results.append(str(file_path))
            file_path = generate_filename(root, 'summary', 'input_with_clusters.xlsx')
            output_df.to_excel(file_path, index=False)
            results.append(str(file_path))       
        else:
            raise AttributeError
    except Exception as e:
        return error(f'Ошибка при сохранении файла с результатом')
    return ready(results)


def run_kmeansscreeplot(params):

   # common
    results = []
    job_id = params['job_id']
    res = validate_input_and_get_dataframe(params['url'], job_id)
    if not res['success']:
        return res
    df = res['dataframe']
    root = get_or_create_dir(config.DOWNLOAD_DIR, job_id)

    # validate params
    res = validate_columns_params(params, len(df.columns))
    if not res['success']:
        return res

    # specific
    columns = parse_columns(params['columns'], len(df.columns))['data']
    
    try:
        data = df.iloc[:, columns]
        distances = []
        krange = list(range(1, params['max_clusters'] + 1))
        for k in krange:
            model = KMeans(n_clusters=k)
            model.fit(data)
            distances.append(model.inertia_)
    except Exception as e:
        return error(f'Ошибка во время работы алгоритма k-средних : {e}')  
   
    try:
        data = df.iloc[:, columns]
        title = f'Каменистая осыпь:\nподбор оптимального k'
        image_format = params['image_format'].lower()
        name = f'Столбцы {", ".join([str(c) for c in columns])}'
        filename = f'{name}.{image_format}'
        file_path = generate_filename(root, 'KmeansScreePlot', filename)
        sns.set()
        fig, ax = plt.subplots()
        sns.lineplot(x=krange, y=distances, markers=['x'], ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Количество кластеров')
        ax.set_ylabel('Сумма квадратов расстояний')
        fig.savefig(file_path, dpi=int(params['image_dpi']), bbox_inches = "tight")
        results.append(str(file_path))
    except Exception as e:
        return error(f'Ошибка при сохранении изображений с результатом : {e}')
        
    return ready(results)

