import os
import uuid
import datetime
import requests
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from flask import Flask, jsonify, request, send_from_directory
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


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
    return params

def run_stats(filename, params):
    results = []
    df = pd.read_csv(uploads / filename)
    data = df.iloc[:, 2:]
    stats = data.describe()
    stats.index.name = "Stats"
    output = 'result.csv'
    if params['transpose']:
        stats.T.to_csv(output)
    else:
        stats.to_csv(output)
    results.append(request.url_root + "downloads/" + output)
    if params['show_graph']:
        sns.set_style("whitegrid")
        sns.set_context("talk")
        for i, column in enumerate(data.columns):
            fig, ax = plt.subplots(2, 1, figsize=(6, 10), dpi=300)
            sns.distplot(data[column], ax=ax[0])
            sns.boxplot(data[column], ax=ax[1])
            figname = f'Column_{i + 1}.png'
            fig.savefig(figname, bbox_inches = 'tight')
            results.append(request.url_root + "downloads/" + figname)
            plt.close(fig)
    return results

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

def run_linear(filename, params):
    results = []
    df = pd.read_csv(uploads / filename)
    data = df.iloc[:, 2:]
    x_column_id = params['x_column'] - 1
    y_column_id = params['y_column'] - 1
    x_column_id = x_column_id if x_column_id < len(data.columns) else 0
    y_column_id = y_column_id if y_column_id < len(data.columns) else 1

    x = data.iloc[:, x_column_id].values.reshape(-1, 1)
    y = data.iloc[:, y_column_id].values.reshape(-1, 1)

    lr = LinearRegression()
    lr.fit(x, y)
    yhat = lr.predict(x)

    res = {
        'Parameter': ['Coefficient', 'Intercept', 'Mean Squared Error'],
        'Value': [lr.coef_[0][0], lr.intercept_[0], mean_squared_error(y, yhat)]
        }
    pd.DataFrame(res).to_csv(output, index=False)
    results.append(request.url_root + "downloads/" + output)
    
    fig, ax = plt.subplots(dpi=300)
    sns.regplot(x, y, yhat, ax=ax)
    ax.set(title=f'y = {lr.coef_[0][0]:.4f}x + {lr.intercept_[0]:.4f}')
    ax.set(xlabel=data.columns[x_column_id], ylabel=data.columns[y_column_id])
    figname = f'Regression_Column{x_column_id}_Column{y_column_id}.png'
    fig.savefig(figname, bbox_inches = 'tight')
    results.append(request.url_root + "downloads/" + figname)

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

@app.route('/stats')
def stats():
    if 'file' in request.args:
        url = request.args['file']
        res = requests.get(url)
        if res.ok:
            name = url.split('/')[-1]
            with open(uploads / name, 'wb') as f:
                f.write(res.content)
            params = parse_params(request.args)
            results = run_stats(name, params)
            return jsonify({'success': True, 'results': results})
    return jsonify({'success': False, 'error': 'File Not Loaded!'})


@app.route('/linear')
def linear():
    if 'file' in request.args:
        url = request.args['file']
        res = requests.get(url)
        if res.ok:
            name = url.split('/')[-1]
            with open(uploads / name, 'wb') as f:
                f.write(res.content)
            params = parse_params(request.args)
            results = run_linear(name, params)
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