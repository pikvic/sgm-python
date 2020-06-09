# TODO
# - функция построения графиков с опциями
# - функция чтения файла в зависимости от формата и выбранных столбцов
# - создание ответа (jsonify...)
# - хранение файлов в редисе какое-то время
# - создание заданий в редисе
# - воркер для заданий в редисе
# - params в keyword агрументы или kwargs
# - в итоге функция подготовки к вызову обработки: валидация, скачивание файла и т.д.

import os
import uuid
import datetime
import requests
from flask import Flask, send_from_directory, jsonify, request
from pathlib import Path
from service import run_stats, run_pca, run_linear, run_kmeans, run_hca


app = Flask(__name__)
root = Path()
uploads = root / 'uploads'
downloads = root / 'downloads'
if not uploads.exists():
    uploads.mkdir()
if not downloads.exists():
    downloads.mkdir()


def parse_params(request_args):
    params = {}
    params['n_clusters'] = int(request_args.get('nclusters', '6'))
    if 'random_state' in request_args:
        params['random_state'] = int(request_args['random_state'])
    else:
        params['random_state'] = None
    params['show_stats'] = bool(int(request_args.get('show_stats', 0)))
    params['add_result_columns'] = bool(int(request_args.get('add_result_columns', 0)))
    params['normalize'] = bool(int(request_args.get('normalize', 0)))
    params['show_graph'] = bool(int(request_args.get('show_graph', 0)))
    params['transpose'] = bool(int(request_args.get('transpose', 0)))
    params['x_column'] = int(request_args.get('x_column', '1'))
    params['y_column'] = int(request_args.get('y_column', '2'))
    params['column'] = int(request_args.get('column', '1'))
    params['exclude'] = request_args.get('exclude', '')
    params['levels'] = request_args.get('levels', '3')
    return params

def check_file_and_run_task(request, task):
    if 'file' in request.args:
        url = request.args['file']
        res = requests.get(url)
        if res.ok:
            name = url.split('/')[-1]
            with open(uploads / name, 'wb') as f:
                f.write(res.content)
            params = parse_params(request.args)
            results = task(name, params)
            if 'error' not in results:
                return jsonify({'success': True, 'results': results})
            else:
                return jsonify(results)
    return jsonify({'success': False, 'error': 'File Not Loaded!'})

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/downloads/<path:filename>')
def get_file(filename):
    if (downloads / filename).exists():
        return send_from_directory(downloads, filename, as_attachment=True, cache_timeout=0)

@app.route('/kmeans')
def kmeans():
    return check_file_and_run_task(request, run_kmeans)
    
@app.route('/pca')
def pca():
    return check_file_and_run_task(request, run_pca)
    
@app.route('/hca')
def hca():
    return check_file_and_run_task(request, run_hca)
    
@app.route('/stats')
def stats():
    return check_file_and_run_task(request, run_stats)
    
@app.route('/linear')
def linear():
    return check_file_and_run_task(request, run_linear)
    
@app.route('/test_file')
def test_file():
    return send_from_directory(Path().cwd(), 'data.csv', as_attachment=True, cache_timeout=0) 

@app.route('/test')
def test():
    return jsonify({'success': True, 'results': ['Test Data']})

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Not found'}), 404


if __name__ == '__main__':
    app.run(threaded=True, port=5000)