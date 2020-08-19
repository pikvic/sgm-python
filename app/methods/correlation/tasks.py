import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

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
    except:
        return error('Ошибка при сохранении файла с результатом')
        
    return ready(results)


def run_corrmatrix(params):
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
    columns1 = parse_columns(params['columns1'], len(df.columns))['data']
    columns2 = parse_columns(params['columns2'], len(df.columns))['data']
    corr = df.corr()
    corr = corr.iloc[columns1, columns2]

    # save output
    try:
        if params['file_format'] == 'CSV':
            file_path = generate_filename(root, 'corrmatrix', 'output.csv')
            corr.to_csv(file_path, index=False)
            results.append(str(file_path))
        elif params['file_format'] == 'XLSX':
            file_path = generate_filename(root, 'corrmatrix', 'output.xlsx')
            corr.to_excel(file_path, index=False)
            results.append(str(file_path))
        else:
            raise AttributeError
    except Exception as e:
        return error(f'Ошибка при сохранении файла с результатом : {e}')
        
    return ready(results)

def run_scatterplot(params):
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
    columns1 = parse_columns(params['columns1'], len(df.columns))['data']
    columns2 = parse_columns(params['columns2'], len(df.columns))['data']

    try:
        for column1 in columns1:
            for column2 in columns2:
                data1 = df.iloc[:, column1]
                data2 = df.iloc[:, column2]
                name1, name2 = data1.name, data2.name
                
                image_format = params['image_format'].lower()
                filename = f'{name1}_{name2}.{image_format}'
                file_path = generate_filename(root, 'scatterplot', filename)
                sns.set()
                fig, ax = plt.subplots()
                p = sns.regplot(data1, data2, line_kws={'color': 'red'}, ax=ax)
                slope, intercept, r_value, p_value, std_err = linregress(x=p.get_lines()[0].get_xdata(),y=p.get_lines()[0].get_ydata())
                title = f'Диаграмма рассеяния\nСтолбцы {name1} и {name2} \ny = {slope:.5f} * x + {intercept:.5f}'
                ax.set_title(title)
                fig.savefig(file_path, dpi=int(params['image_dpi']), bbox_inches = "tight")
                results.append(str(file_path))
    except Exception as e:
        return error(f'Ошибка при сохранении изображений с результатом : {e}')
        
    return ready(results)