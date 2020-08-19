import re
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from .schema import StrategyEnum

import app.core.config as config
from app.core.tasks import validate_input_and_get_dataframe, get_or_create_dir, generate_filename, error, ready, validate_columns_params, parse_columns


def run_missingvalues(params):
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

    try:
        if params['strategy'] == StrategyEnum.MEAN:
            strategy = 'mean'
        elif params['strategy'] == StrategyEnum.MEDIAN:
            strategy = 'median'
        elif params['strategy'] == StrategyEnum.MOST_FREQUENT:
            strategy = 'most_frequent'
        elif params['strategy'] == StrategyEnum.CONSTANT:
            strategy = 'constant'

        imp = SimpleImputer(missing_values=params['missing_value'], strategy='mean', fill_value=params['fill_value'])
        result = imp.fit_transform(data)

        for i, column in enumerate(columns):
            df.iloc[:, column] = result[:, i]
    except Exception as e:
        return error(f'Ошибка при выполнении замены пропущенных значений: {e}')

    # save output
    try:
        if params['file_format'] == 'CSV':
            file_path = generate_filename(root, 'missingvalues', 'input_replaced.csv')
            df.to_csv(file_path, index=False)
            results.append(str(file_path))
        elif params['file_format'] == 'XLSX':
            file_path = generate_filename(root, 'missingvalues', 'input_replaced.xlsx')
            df.to_excel(file_path, index=False)
            results.append(str(file_path))
        else:
            raise AttributeError
    except Exception as e:
        return error(f'Ошибка при сохранении файла с результатом : {e}')
        
    return ready(results)


def run_normalization(params):

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

    try:
        scaler = MinMaxScaler(feature_range=(params['lower_bound'], params['upper_bound']))
        result = scaler.fit_transform(data)
        for i, column in enumerate(columns):
            df.iloc[:, column] = result[:, i]
    except Exception as e:
        return error(f'Ошибка при выполнении нормализации: {e}')

    # save output
    try:
        if params['file_format'] == 'CSV':
            file_path = generate_filename(root, 'normalization', 'input_replaced.csv')
            df.to_csv(file_path, index=False)
            results.append(str(file_path))
        elif params['file_format'] == 'XLSX':
            file_path = generate_filename(root, 'normalization', 'input_replaced.xlsx')
            df.to_excel(file_path, index=False)
            results.append(str(file_path))
        else:
            raise AttributeError
    except Exception as e:
        return error(f'Ошибка при сохранении файла с результатом : {e}')
        
    return ready(results)

def run_standartization(params):
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

    try:
        scaler = StandardScaler()
        result = scaler.fit_transform(data)
        for i, column in enumerate(columns):
            df.iloc[:, column] = result[:, i]
    except Exception as e:
        return error(f'Ошибка при выполнении стандартизации: {e}')

    # save output
    try:
        if params['file_format'] == 'CSV':
            file_path = generate_filename(root, 'standartization', 'input_replaced.csv')
            df.to_csv(file_path, index=False)
            results.append(str(file_path))
        elif params['file_format'] == 'XLSX':
            file_path = generate_filename(root, 'standartization', 'input_replaced.xlsx')
            df.to_excel(file_path, index=False)
            results.append(str(file_path))
        else:
            raise AttributeError
    except Exception as e:
        return error(f'Ошибка при сохранении файла с результатом : {e}')
        
    return ready(results)