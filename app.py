import os
import uuid
import datetime
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from flask import Flask, jsonify, request, send_from_directory
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
    return params

def run_kmeans(filename, params):
    results = []
    df = pd.read_csv(uploads / filename)
    data = df.iloc[:, 2:]
    if params['normalize']:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=params['n_clusters'], random_state=params['random_state'])
    labels = kmeans.fit_predict(data) + 1
    if params['add_result_columns']:
        df['Cluster'] = labels
        df.to_csv(downloads / 'kmeans_result.csv', index=False)
    else:
        new_df = pd.DataFrame(labels, columns=['Cluster'])
        new_df.to_csv(downloads / 'kmeans_result.csv', index=False)
    results.append(request.url_root + "downloads/" + 'kmeans_result.csv')
    if params['show_stats']:
        centers = kmeans.cluster_centers_
        inertia = kmeans.inertia_
    if params['show_graph']:
        pca = PCA(n_components=2)
        components = pca.fit_transform(data)
        fig, ax = plt.subplots()
        scatter = ax.scatter(components[:, 0], components[:, 1], c=labels)
        ax.set_title('Clustering In Principal Components')
        ax.set_xlabel('Principal Component 0')
        ax.set_ylabel('Principal Component 1')
        ax.legend(*scatter.legend_elements(), title="Clusters")
        fig.savefig(downloads / 'kmeans_figure.png', dpi=300)
        results.append(request.url_root + "downloads/" + 'kmeans_figure.png')
    return results

def run_pca(filename, params):
    results = []
    df = pd.read_csv(uploads / filename)
    data = df.iloc[:, 2:]
    params['show_graph'] = True
    params['normalize'] = True
    if params['normalize']:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    pca = PCA()
    components = pca.fit_transform(data)
    if params['show_graph']:
        fig, ax = plt.subplots()
        ax.bar(list(range(1, pca.n_components_ + 1)), pca.explained_variance_ratio_)
        ax.set_title('Principal Component Analysis')
        ax.set_xlabel('Number of components')
        ax.set_ylabel('Explained Variance Ratio')
        fig.savefig(downloads / 'pca_figure_1.png', dpi=300)
        results.append(request.url_root + "downloads/" + 'pca_figure_1.png')
        fig, ax = plt.subplots()
        ax.plot(list(range(1, pca.n_components_ + 1)), np.cumsum(pca.explained_variance_ratio_))
        ax.set_title('Principal Component Analysis')
        ax.set_xlabel('Number of components')
        ax.set_ylabel('Cumulative Explained Variance Ratio')
        fig.savefig(downloads / 'pca_figure_2.png', dpi=300)
        results.append(request.url_root + "downloads/" + 'pca_figure_2.png')
    return results

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/downloads/<path:filename>')
def get_file(filename):
    if (downloads / filename).exists():
        return send_from_directory(downloads, filename, as_attachment=True, cache_timeout=0)

@app.route('/kmeans')
def kmeans():
    if 'file' in request.args:
        url = request.args['file']
        res = requests.get(url)
        if res.ok:
            name = url.split('/')[-1]
            with open(uploads / name, 'wb') as f:
                f.write(res.content)
            params = parse_params(request.args)
            results = run_kmeans(name, params)
            return jsonify({'success': True, 'results': results})
    return jsonify({'success': False, 'error': 'File Not Loaded!'})

@app.route('/pca')
def pca():
    if 'file' in request.args:
        url = request.args['file']
        res = requests.get(url)
        if res.ok:
            name = url.split('/')[-1]
            with open(uploads / name, 'wb') as f:
                f.write(res.content)
            params = parse_params(request.args)
            results = run_pca(name, params)
            return jsonify({'success': True, 'results': results})
    return jsonify({'success': False, 'error': 'File Not Loaded!'})


@app.route('/test')
def test():
    return jsonify({'success': True, 'results': ['Test Data']})

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Not found'}), 404


if __name__ == '__main__':
    app.run(threaded=True, port=5000)