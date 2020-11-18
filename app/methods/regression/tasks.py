import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

import app.core.config as config
from app.core.tasks import validate_input_and_get_dataframe, get_or_create_dir, generate_filename, error, ready, validate_columns_params, parse_columns


def run_linearregression(params):
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
    target = df.iloc[:, params['target_column']]

    try:
        lr = LinearRegression()
        lr.fit(data, target)
        coef = {}
        for i, name in enumerate(data.columns):
            name = f'Переменная {name}'
            coef[name] = [lr.coef_[i]]
            coef["Свободный член"] = [lr.intercept_]
            coef["Коэффициент детерминации"] = [lr.score(data, target)]
        linear_df = pd.DataFrame(coef.values(), index=coef.keys(), columns=['Значение'])
    except:
        return error('Ошибка при вычислении результата')

    # save output
    try:
        if params['file_format'] == 'CSV':
            file_path = generate_filename(root, 'linearregression', 'coefficients.csv')
            linear_df.to_csv(file_path, index_label='Переменная')
            results.append(str(file_path))
        elif params['file_format'] == 'XLSX':
            file_path = generate_filename(root, 'linearregression', 'coefficients.xlsx')
            linear_df.to_excel(file_path, index_label='Переменная')
            results.append(str(file_path))
        else:
            raise AttributeError
    except:
        return error('Ошибка при сохранении файла с результатом')

    return ready(results)

def run_polynomialregression(params):
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
    target = df.iloc[:, params['target_column']]    

    try:
        polyreg = make_pipeline(PolynomialFeatures(params['degree']),LinearRegression())
        polyreg.fit(data, target)
        index = []
        for power in polyreg[0].powers_:
            strings = []
            for i, column in enumerate(data.columns):
                strings.append(f'{column}^{power[i]}')
            string = ' * '.join(strings)
            index.append(string)
        poly_df = pd.DataFrame(polyreg[1].coef_, index=index, columns=['Значение коэффициента'])    
    except:
        return error('Ошибка при вычислении результата')

    # save output
    try:
        if params['file_format'] == 'CSV':
            file_path = generate_filename(root, 'polynomialregression', 'coefficients.csv')
            poly_df.to_csv(file_path, index_label='Слагаемое')
            results.append(str(file_path))
        elif params['file_format'] == 'XLSX':
            file_path = generate_filename(root, 'polynomialregression', 'coefficients.xlsx')
            poly_df.to_excel(file_path, index_label='Слагаемое')
            results.append(str(file_path))
        else:
            raise AttributeError
    except:
        return error('Ошибка при сохранении файла с результатом')
   
    return ready(results)