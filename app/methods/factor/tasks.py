import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.decomposition import PCA, FactorAnalysis

import app.core.config as config
from app.core.tasks import validate_input_and_get_dataframe, get_or_create_dir, generate_filename, error, ready, validate_columns_params, parse_columns


def run_factoranalysis(params):
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
        fa_model = FactorAnalysis(n_components=params['ncomponents'])
        fa_model.fit(data)
        factor_data = fa_model.transform(data)
        loading = fa_model.components_
        loading_df = pd.DataFrame(loading, index=list(range(1, params['ncomponents'] + 1)), columns=data.columns)
        factor_data_df = pd.DataFrame(factor_data, columns=[f"Фактор {i}" for i in range(1, params['ncomponents'] + 1)])
    except:
        return error('Ошибка при вычислении результата')

    try:
        data = df.iloc[:, columns]
        image_format = params['image_format'].lower()
        component_names = data.columns
        
        for i in range(1, params['ncomponents'] + 1):
            filename = f'Factor_{i}_Loadings.{image_format}'
            file_path = generate_filename(root, 'factoranalysis', filename)
            sns.set()
            fig, ax = plt.subplots()
            sns.barplot(loading[i-1], component_names, orient='h', ax=ax)
            ax.set_title(f'Факторная нагрузка для фактора {i}')
            ax.set_xlabel('Факторная нагрузка')
            ax.set_ylabel('Переменная')
            fig.savefig(file_path, dpi=int(params['image_dpi']), bbox_inches = "tight")
            results.append(str(file_path))

    except Exception as e:
        return error(f'Ошибка при сохранении изображений с результатом : {e}')
          
    # save output
    try:
        if params['file_format'] == 'CSV':
            file_path = generate_filename(root, 'factoranalysis', 'factorized.csv')
            factor_data_df.to_csv(file_path, index=False)
            results.append(str(file_path))
        elif params['file_format'] == 'XLSX':
            file_path = generate_filename(root, 'factoranalysis', 'factorized.xlsx')
            factor_data_df.to_excel(file_path, index=False)
            results.append(str(file_path))
        else:
            raise AttributeError
    except:
        return error('Ошибка при сохранении файла с результатом')

    # save output
    try:
        if params['file_format'] == 'CSV':
            file_path = generate_filename(root, 'factoranalysis', 'loadings.csv')
            loading_df.to_csv(file_path, index_label='Номер фактора')
            results.append(str(file_path))
        elif params['file_format'] == 'XLSX':
            file_path = generate_filename(root, 'factoranalysis', 'loadings.xlsx')
            loading_df.to_excel(file_path, index_label='Номер фактора')
            results.append(str(file_path))
        else:
            raise AttributeError
    except:
        return error('Ошибка при сохранении файла с результатом')

    return ready(results)



def run_factorscreeplot(params):
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
        covar_matrix = np.cov(data, rowvar=False)
        eigenvalues = np.linalg.eig(covar_matrix)[0]
    except:
        return error('Ошибка при вычислении результата')


    try:
        data = df.iloc[:, columns]
        image_format = params['image_format'].lower()
        filename = f'plot.{image_format}'
        file_path = generate_filename(root, 'factorscreeplot', filename)
        
        sns.set()
        fig, ax = plt.subplots()
        x = list(range(1, len(eigenvalues) + 1))
        sns.lineplot(x, eigenvalues, marker='o', ax=ax)
        ax.axhline(y=1.0, color='r', linestyle='--')
        ax.set_xticks(x[::2]) # <--- set the ticks first
        ax.set_title('Определение количества факторов (каменистая осыпь)')
        ax.set_xlabel('Количество факторов')
        ax.set_ylabel('Собственное значение')

        fig.savefig(file_path, dpi=int(params['image_dpi']), bbox_inches = "tight")
        results.append(str(file_path))
    except Exception as e:
        return error(f'Ошибка при сохранении изображений с результатом : {e}')
      
    return ready(results)


def run_pca(params):
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
        pca = PCA(n_components=params['ncomponents'])
        components = pca.fit_transform(data)
        components_stats_df = pd.DataFrame([pca.explained_variance_, pca.explained_variance_ratio_, pca.singular_values_])
        components_stats_df.columns = ["Дисперсии осей проекции (выборочная)", "Доля информации (доля от общей дисперсии)", "Сингулярное значение"]
        components_df = pd.DataFrame(pca.components_)
        components_df.columns = data.columns
    except:
        return error('Ошибка при вычислении результата')

    # save output
    try:
        if params['file_format'] == 'CSV':
            file_path = generate_filename(root, 'pca', 'components_stats.csv')
            components_stats_df.to_csv(file_path, index_label='Номер компоненты')
            results.append(str(file_path))
        elif params['file_format'] == 'XLSX':
            file_path = generate_filename(root, 'pca', 'components_stats.xlsx')
            components_stats_df.to_excel(file_path, index_label='Номер компоненты')
            results.append(str(file_path))
        else:
            raise AttributeError
    except:
        return error('Ошибка при сохранении файла с результатом')
    
    # save output
    try:
        if params['file_format'] == 'CSV':
            file_path = generate_filename(root, 'pca', 'components.csv')
            components_df.to_csv(file_path, index_label='Номер компоненты')
            results.append(str(file_path))
        elif params['file_format'] == 'XLSX':
            file_path = generate_filename(root, 'pca', 'components.xlsx')
            components_df.to_excel(file_path, index_label='Номер компоненты')
            results.append(str(file_path))
        else:
            raise AttributeError
    except:
        return error('Ошибка при сохранении файла с результатом')

    return ready(results)

def run_pcachoose(params):
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
        image_format = params['image_format'].lower()
        filename = f'pca_choose_number_of_components.{image_format}'
        file_path = generate_filename(root, 'pcachoose', filename)
        
        pca = PCA()
        components = pca.fit_transform(data)
        sns.set()
        fig, ax = plt.subplots()
        x = list(range(1, pca.n_components_ + 1))
        y = np.cumsum(pca.explained_variance_ratio_)
        p = sns.lineplot(x, y, marker='o', ax=ax)
        p.set_xticks(x[::2]) # <--- set the ticks first
        ax.set_title('Анализ главных компонент')
        ax.set_xlabel('Количество компонент')
        ax.set_ylabel('Доля информации (доля от общей дисперсии)')
        fig.savefig(file_path, dpi=int(params['image_dpi']), bbox_inches = "tight")
        results.append(str(file_path))
    except Exception as e:
        return error(f'Ошибка при сохранении изображений с результатом : {e}')
      
    return ready(results)


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
            corr.to_csv(file_path)
            results.append(str(file_path))
        elif params['file_format'] == 'XLSX':
            file_path = generate_filename(root, 'corrmatrix', 'output.xlsx')
            corr.to_excel(file_path)
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