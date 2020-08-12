import re
import requests
import pandas as pd
import app.core.config as config


def clear_files_for_job(job_id):
    path = config.UPLOAD_DIR / job_id
    if path.exists():
        for f in path.iterdir():
            f.unlink()
        path.rmdir()
    path = config.DOWNLOAD_DIR / job_id
    if path.exists():
        for f in path.iterdir():
            f.unlink()
        path.rmdir()

def get_or_create_dir(root, job_id):
    path = root / job_id
    if not path.exists():
        path.mkdir()
    return path

def upload_file(url, job_id):
    try:
        root = get_or_create_dir(config.UPLOAD_DIR, job_id)
        res = requests.get(url)
        if res.ok:
            name = url.split('/')[-1]
            with open(root / name, 'wb') as f:
                f.write(res.content)
    except Exception as e:
        return {'success': False, 'error': f'Невозможно скачать файл {url}'}
    return {'success': True, 'file_path': f'{root / name}'}

def check_file(filename):
    error = False
    file_format = None
    try:
        df = pd.read_csv(filename)
        error = False
        file_format = 'csv'
    except Exception as e:
        error = True
    if error:
        try:
            df = pd.read_excel(filename)
            error = False
            file_format = 'excel'
        except Exception as e:
            error = True 
    if error:
        return {'success': False, 'error': 'Неверный формат входного файла'}
    return {'success': True, 'file_format': file_format}

def get_dataframe(filename, file_format):
    try:
        if file_format == 'excel':
            df = pd.read_excel(filename)
        elif file_format == 'csv':
            df = pd.read_csv(filename)
    except:
        return {'success': False, 'error': 'Неправильная структура таблиц в файле'}
    return {'success': True, 'dataframe': df}

def validate_input_and_get_dataframe(url, job_id):
    res = upload_file(url, job_id)
    if not res['success']:
        return res
    filename = res['file_path']
    res = check_file(filename)
    if not res['success']:
        return res
    res = get_dataframe(filename, res['file_format'])
    if not res['success']:
        return res
    return res

def parse_columns(param, ncolumns):
    pattern = r'(^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$)|(^\*$)'
    pattern_all = r'^\*$'
    if not re.search(pattern, param):
        return {'success': False, 'error': 'Неверный формат указания столбцов для обработки'}
    if re.search(pattern_all, param):
        return {'success': True, 'data': list(range(ncolumns))}
    
    res = re.findall(r'\d+(?:-\d+)*', param)
    columns = set()
    for r in res:
        if '-' in r:
            left, right = int(r.split('-')[0]), int(r.split('-')[1])
            if left < right:
                if left > ncolumns or right > ncolumns:
                    return {'success': False, 'error': 'Указанные номера столбцов больше, чем есть во входном файле'}
                columns = columns | set(range(left - 1, right))
        else:
            col = int(r)
            if col > ncolumns:
                return {'success': False, 'error': 'Указанные номера столбцов больше, чем есть во входном файле'}

            columns.add(col - 1)
    columns = sorted(list(columns))
    return {'success': True, 'data': columns}

def validate_columns_params(params, ncolumns):
    if 'columns' in params:
        res = parse_columns(params['columns'], ncolumns)
        if not res['success']:
            return res
    if 'columns1' in params:
        res = parse_columns(params['columns1'], ncolumns)
        if not res['success']:
            return res
    if 'columns2' in params:
        res = parse_columns(params['columns2'], ncolumns)
        if not res['success']:
            return res
    return {'success': True}

def generate_filename(path, prefix, name):
    return path / f'{prefix}_{name}'

def error(message):
    return {'success': False, 'error': message}

def ready(results):
    return {'ready': True, 'results': results}
    
