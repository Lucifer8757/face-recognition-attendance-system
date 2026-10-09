import os
import io
import base64
import urllib.request
import logging
import cv2
import numpy as np
from config import Config

logger = logging.getLogger(__name__)

class FaceRecognitionService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FaceRecognitionService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.ensure_models()
        self.detector = None
        self.recognizer = None
        self.haar_cascade = None
        self._load_models()
        self._initialized = True

    def ensure_models(self):
        """Ensures that required ONNX and XML models exist, downloading if missing."""
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        
        models_to_fetch = {
            Config.YUNET_MODEL: "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            Config.SFACE_MODEL: "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
            Config.HAAR_CASCADE: "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml"
        }
        
        for path, url in models_to_fetch.items():
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                logger.info(f"Downloading required model: {os.path.basename(path)}...")
                try:
                    urllib.request.urlretrieve(url, path)
                    logger.info(f"Model saved to {path}")
                except Exception as e:
                    logger.error(f"Failed to download {path} from {url}: {e}")

    def _load_models(self):
        """Loads YuNet detector and SFace recognizer."""
        try:
            if os.path.exists(Config.YUNET_MODEL) and os.path.exists(Config.SFACE_MODEL):
                # Default input size (320, 320), dynamically updated during detection
                self.detector = cv2.FaceDetectorYN.create(
                    Config.YUNET_MODEL,
                    "",
                    (320, 320),
                    score_threshold=0.6,
                    nms_threshold=0.3,
                    top_k=5000
                )
                self.recognizer = cv2.FaceRecognizerSF.create(Config.SFACE_MODEL, "")
                logger.info("Successfully loaded YuNet and SFace models.")
            else:
                logger.warning("YuNet or SFace model files not found!")
        except Exception as e:
            logger.error(f"Error loading YuNet / SFace: {e}")

        # Load Haar Cascade as auxiliary detector if available in cv2
        if hasattr(cv2, 'CascadeClassifier') and os.path.exists(Config.HAAR_CASCADE):
            try:
                self.haar_cascade = cv2.CascadeClassifier(Config.HAAR_CASCADE)
            except Exception as e:
                logger.warning(f"Could not load Haar cascade: {e}")

    def decode_base64_image(self, b64_string):
        """Converts base64 data URI or raw base64 string to OpenCV BGR image."""
        if not b64_string:
            return None
        if ',' in b64_string:
            b64_string = b64_string.split(',', 1)[1]
        try:
            img_bytes = base64.b64decode(b64_string)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            return None

    def encode_image_base64(self, image_np, format_ext='.jpg'):
        """Converts OpenCV BGR image to base64 data URI."""
        if image_np is None:
            return None
        try:
            _, buffer = cv2.imencode(format_ext, image_np)
            b64_data = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{b64_data}"
        except Exception as e:
            logger.error(f"Failed to encode image to base64: {e}")
            return None

    def detect_faces(self, image_np):
        """
        Detects faces in image using YuNet.
        Returns list of dicts:
        [{
            'box': [x, y, w, h],
            'confidence': float,
            'landmarks': [[x,y], ...],
            'raw_face': numpy array
        }]
        """
        if image_np is None:
            return []

        h, w = image_np.shape[:2]
        detected = []

        if self.detector is not None:
            try:
                self.detector.setInputSize((w, h))
                _, faces = self.detector.detect(image_np)
                if faces is not None:
                    for f in faces:
                        box = [int(f[0]), int(f[1]), int(f[2]), int(f[3])]
                        # Landmark coordinates (right eye, left eye, nose tip, right mouth, left mouth)
                        landmarks = [
                            [float(f[4]), float(f[5])],
                            [float(f[6]), float(f[7])],
                            [float(f[8]), float(f[9])],
                            [float(f[10]), float(f[11])],
                            [float(f[12]), float(f[13])]
                        ]
                        score = float(f[14])
                        detected.append({
                            'box': box,
                            'confidence': score,
                            'landmarks': landmarks,
                            'raw_face': f
                        })
                    return detected
            except Exception as e:
                logger.error(f"YuNet detection error: {e}")

        # Fallback to Haar cascade if YuNet had no results or is unavailable
        if len(detected) == 0 and self.haar_cascade is not None:
            try:
                gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
                haar_faces = self.haar_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )
                for (x, y, fw, fh) in haar_faces:
                    # Synthetic raw_face format compatible with alignCrop
                    # Approximate landmarks based on face box proportions
                    re = [x + fw * 0.3, y + fh * 0.35]
                    le = [x + fw * 0.7, y + fh * 0.35]
                    nt = [x + fw * 0.5, y + fh * 0.55]
                    rc = [x + fw * 0.35, y + fh * 0.75]
                    lc = [x + fw * 0.65, y + fh * 0.75]
                    raw = np.array([x, y, fw, fh, re[0], re[1], le[0], le[1], nt[0], nt[1], rc[0], rc[1], lc[0], lc[1], 0.85], dtype=np.float32)
                    detected.append({
                        'box': [int(x), int(y), int(fw), int(fh)],
                        'confidence': 0.85,
                        'landmarks': [re, le, nt, rc, lc],
                        'raw_face': raw
                    })
            except Exception as e:
                logger.error(f"Haar cascade detection error: {e}")

        return detected

    def extract_embedding(self, image_np, raw_face=None):
        """
        Extracts 128-dimensional embedding from image.
        If raw_face is provided, uses landmark-based alignment;
        otherwise runs detection first.
        Returns numpy float32 array of shape (128,) or None.
        """
        if image_np is None:
            return None

        if raw_face is None:
            faces = self.detect_faces(image_np)
            if not faces:
                return None
            raw_face = faces[0]['raw_face']

        try:
            if self.recognizer is not None:
                aligned = self.recognizer.alignCrop(image_np, raw_face)
                feat = self.recognizer.feature(aligned)
                # Flatten to 1D (128,) float32 vector
                embedding = feat.flatten().astype(np.float32)
                # Normalize vector to unit length
                norm = np.linalg.norm(embedding)
                if norm > 1e-6:
                    embedding = embedding / norm
                return embedding
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None

    def compare_embeddings(self, emb1, emb2):
        """
        Computes cosine similarity between two 128D embeddings.
        Returns float between -1.0 and 1.0. Higher is more similar.
        """
        if emb1 is None or emb2 is None:
            return 0.0
        try:
            # emb1, emb2 are normalized 1D vectors
            v1 = np.asarray(emb1, dtype=np.float32).flatten()
            v2 = np.asarray(emb2, dtype=np.float32).flatten()
            dot = float(np.dot(v1, v2))
            norm1 = float(np.linalg.norm(v1))
            norm2 = float(np.linalg.norm(v2))
            if norm1 * norm2 == 0:
                return 0.0
            return dot / (norm1 * norm2)
        except Exception as e:
            logger.error(f"Error computing cosine similarity: {e}")
            return 0.0

    def find_best_match(self, query_emb, enrolled_list, threshold=0.38):
        """
        Compares query embedding with enrolled embeddings list.
        enrolled_list is a list of tuples: (student_id, student_name, roll_number, embedding_vec)
        Returns:
            (best_student_info_dict or None, best_score, score_percentage)
        """
        if query_emb is None or not enrolled_list:
            return None, 0.0, 0.0

        best_score = -1.0
        best_match = None

        for student_id, name, roll_no, emb in enrolled_list:
            score = self.compare_embeddings(query_emb, emb)
            if score > best_score:
                best_score = score
                best_match = {
                    'student_id': student_id,
                    'full_name': name,
                    'roll_number': roll_no,
                    'score': float(score)
                }

        # Percentage representation for friendly UI display
        score_percent = max(0.0, min(100.0, float(best_score * 100)))

        if best_match and best_score >= threshold:
            best_match['match_percentage'] = round(score_percent, 1)
            return best_match, best_score, score_percent

        return None, best_score, score_percent

    def check_liveness_heuristic(self, image_np, box=None):
        """
        Basic passive liveness heuristic:
        Computes image sharpness (Laplacian variance) to flag blurry screens or printed paper artifacts.
        Note: Clearly explained in privacy & limitations section as basic 2D check.
        """
        if image_np is None:
            return False, 0.0
        try:
            if box:
                x, y, w, h = box
                h_img, w_img = image_np.shape[:2]
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(w_img, x + w)
                y2 = min(h_img, y + h)
                crop = image_np[y1:y2, x1:x2]
            else:
                crop = image_np

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            # If variance is too low (< 50), face is likely blurry or out of focus
            is_clear = bool(variance > 45.0)
            return is_clear, float(variance)
        except Exception as e:
            logger.error(f"Liveness check error: {e}")
            return True, 100.0
