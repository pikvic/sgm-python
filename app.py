import os
import uuid
import requests
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from flask import Flask, jsonify, request, send_file


app = Flask(__name__)
root = Path()
uploads = root / 'uploads'
downloads = root / 'downloads'
if not uploads.exists():
    uploads.mkdir()

if not downloads.exists():
    downloads.mkdir()


def run_kmeans(filename, n_clusters=6):
    df = pd.read_csv(uploads / filename)
    data = df.iloc[:, 2:]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(data)
    df['Cluster'] = kmeans.labels_ + 1
    df.to_csv(downloads / filename)


@app.route('/')
def index():
    return "Hello, World!"


@app.route('/downloads/<filename>')
def get_file(filename):
    if (downloads / filename).exists():
        return send_file(downloads / filename)


@app.route('/kmeans')
def kmeans():
    if 'file' in request.args:
        url = request.args['file']
        res = requests.get(url)
        if res.ok:
            name = url.split('/')[-1]
            with open(uploads / name, 'wb') as f:
                f.write(res.content)
            nclusters = request.args.get('nclusters', 6)
            run_kmeans(name, n_clusters=nclusters)
            return jsonify({'success': True, 'result': request.url_root + "downloads/" + str(name)})
    return jsonify({'success': False, 'error': 'File Not Loaded!'})


@app.route('/test')
def test():
    return jsonify({'success': True, 'result': 'Test Data'})


@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Not found'}), 404


if __name__ == '__main__':
    app.run(threaded=True, port=5000)