import re
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import app.core.config as config
from app.core.tasks import validate_input_and_get_dataframe, get_or_create_dir, generate_filename, error, ready, validate_columns_params, parse_columns


def run_summary(params):
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
    data = df.iloc[:, columns]
    stats = data.describe().T
    stats.index.name = "Параметр"
    stats.columns = ['Количество', 'Среднее', 'Стандартное отклонение', 'Минимум', '25й перцентиль', 'Медиана', '75% перцентиль', 'Максимум']

    # save output
    try:
        if params['file_format'] == 'CSV':
            file_path = generate_filename(root, 'summary', 'output.csv')
            stats.to_csv(file_path)
            results.append(str(file_path))
        elif params['file_format'] == 'XLSX':
            file_path = generate_filename(root, 'summary', 'output.xlsx')
            stats.to_excel(file_path)
            results.append(str(file_path))
        else:
            raise AttributeError
    except Exception as e:
        return error(f'Ошибка при сохранении файла с результатом')
        
    return ready(results)


def run_histogram(params):

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
        for column in columns:
            data = df.iloc[:, column]
            name = data.name
            title = f'Гистограмма распределения\nСтолбец {name}'
            image_format = params['image_format'].lower()
            filename = f'{name}.{image_format}'
            file_path = generate_filename(root, 'histogram', filename)
            sns.set()
            fig, ax = plt.subplots()
            sns.distplot(data, ax=ax)
            ax.set_title(title)
            fig.savefig(file_path, dpi=int(params['image_dpi']))
            results.append(str(file_path))
    except Exception as e:
        return error(f'Ошибка при сохранении изображений с результатом : {e}')
        
    return ready(results)

def run_boxplot(params):
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
        for column in columns:
            data = df.iloc[:, column]
            name = data.name
            title = f'Диаграмма размаха\nСтолбец {name}'
            image_format = params['image_format'].lower()
            filename = f'{name}.{image_format}'
            file_path = generate_filename(root, 'boxplot', filename)
            sns.set()
            fig, ax = plt.subplots()
            sns.boxplot(width=0.5, x=data, ax=ax)
            ax.set_title(title)
            fig.savefig(file_path, dpi=int(params['image_dpi']))
            results.append(str(file_path))
    except Exception as e:
        return error(f'Ошибка при сохранении изображений с результатом : {e}')
        
    return ready(results)