import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
IS_VERCEL = bool(os.environ.get('VERCEL'))

class Config:
    BASE_DIR = BASE_DIR
    SECRET_KEY = os.environ.get('SECRET_KEY', 'smart-attendance-system-secret-key-2026')
    
    # Use /tmp on Vercel serverless environment (read-only filesystem workaround)
    DATA_DIR = os.path.join('/tmp', 'data') if IS_VERCEL else os.path.join(BASE_DIR, 'data')
    DATABASE_PATH = os.path.join(DATA_DIR, 'face_attendance.db')
    
    # Secure storage (not inside static/ for privacy and security)
    SECURE_STORAGE_DIR = os.path.join(DATA_DIR, 'secure_storage')
    PHOTOS_DIR = os.path.join(SECURE_STORAGE_DIR, 'photos')
    EMBEDDINGS_DIR = os.path.join(SECURE_STORAGE_DIR, 'embeddings')
    
    # Models directory
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    YUNET_MODEL = os.path.join(MODELS_DIR, 'yunet.onnx')
    SFACE_MODEL = os.path.join(MODELS_DIR, 'sface.onnx')
    HAAR_CASCADE = os.path.join(MODELS_DIR, 'haarcascade_frontalface_default.xml')
    
    # Face recognition parameters
    DEFAULT_SIMILARITY_THRESHOLD = 0.38  # Cosine similarity threshold for SFace
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # Max 5MB upload
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
    
    # Admin default credentials
    DEFAULT_ADMIN_USERNAME = os.environ.get('ADMIN_USER', 'admin')
    DEFAULT_ADMIN_PASSWORD = os.environ.get('ADMIN_PASS', 'admin123')
