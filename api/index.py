import os
import sys

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ['VERCEL'] = '1'

from app import app
from database import init_db
from database.seed import seed_database
from config import Config

# Cold boot check for Vercel Serverless environment
# Populate initial schema and sample students into /tmp/data if not present
if not os.path.exists(Config.DATABASE_PATH):
    try:
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        os.makedirs(Config.PHOTOS_DIR, exist_ok=True)
        os.makedirs(Config.EMBEDDINGS_DIR, exist_ok=True)
        init_db()
        seed_database(include_demo_attendance=True)
    except Exception as e:
        print(f"Vercel DB initialization notice: {e}")

# Handler exposed for Vercel WSGI
app = app
