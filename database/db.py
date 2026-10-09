import sqlite3
import os
from werkzeug.security import generate_password_hash
from config import Config

def get_db_connection():
    """Establishes a connection to the SQLite database with Row factory."""
    os.makedirs(os.path.dirname(Config.DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    """Initializes database schema and sets default admin and system settings."""
    os.makedirs(Config.PHOTOS_DIR, exist_ok=True)
    os.makedirs(Config.EMBEDDINGS_DIR, exist_ok=True)
    
    schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema_sql = f.read()

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.executescript(schema_sql)
    
    # Insert default admin if not exists
    cursor.execute("SELECT id FROM admin_users WHERE username = ?", (Config.DEFAULT_ADMIN_USERNAME,))
    if not cursor.fetchone():
        hashed_pw = generate_password_hash(Config.DEFAULT_ADMIN_PASSWORD)
        cursor.execute(
            "INSERT INTO admin_users (username, password_hash, full_name, role) VALUES (?, ?, ?, ?)",
            (Config.DEFAULT_ADMIN_USERNAME, hashed_pw, 'System Administrator', 'admin')
        )
    
    # Insert default system settings
    default_settings = [
        ('similarity_threshold', str(Config.DEFAULT_SIMILARITY_THRESHOLD), 'Cosine similarity threshold for face recognition'),
        ('liveness_detection', '0', 'Enable anti-spoofing / liveness challenge (0=disabled, 1=enabled)'),
        ('institution_name', "Government Women's Polytechnic, Ranchi", 'Name of the educational institution'),
        ('academic_year', '2026', 'Current academic year'),
        ('attendance_window_start', '08:00', 'Daily attendance opening time'),
        ('attendance_window_end', '17:00', 'Daily attendance closing time')
    ]
    for key, val, desc in default_settings:
        cursor.execute(
            "INSERT OR IGNORE INTO system_settings (key, value, description) VALUES (?, ?, ?)",
            (key, val, desc)
        )
        
    conn.commit()
    conn.close()

def get_setting(key, default=None):
    """Retrieves a setting value by key."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM system_settings WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    return row['value'] if row else default

def update_setting(key, value):
    """Updates or inserts a setting value."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO system_settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value))
    )
    conn.commit()
    conn.close()
