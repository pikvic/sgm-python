from flask import Flask, jsonify, request
from pathlib import Path
import os


app = Flask(__name__)
root = Path()
uploads = root / 'uploads'
if not uploads.exists():
    uploads.mkdir()


@app.route('/')
def index():
    return "Hello, World!"


@app.route('/test')
def test():
    return jsonify({'success': True, 'result': 'Test Data'})


@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Not found'}), 404


if __name__ == '__main__':
    app.run(threaded=True, port=5000)