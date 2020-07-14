import os
from pathlib import Path

ROOT =  Path()
UPLOAD_DIR = ROOT / 'upload'
DOWNLOAD_DIR = ROOT / 'download'
RESULT_TTL = 600
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
