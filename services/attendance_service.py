import os
import datetime
import io
import json
import logging
import sqlite3
import cv2
import numpy as np
import pandas as pd
from config import Config
from database import get_db_connection, get_setting
from services.face_recognition_service import FaceRecognitionService

logger = logging.getLogger(__name__)

class AttendanceService:
    def __init__(self):
        self.face_service = FaceRecognitionService()

    def get_enrolled_students_with_embeddings(self):
        """
        Loads all registered students who have face embeddings.
        Returns list of tuples: (student_id, full_name, roll_number, embedding_vec)
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        query = """
            SELECT s.student_id, s.full_name, s.roll_number, fe.embedding_vector
            FROM students s
            JOIN face_embeddings fe ON s.student_id = fe.student_id
            WHERE s.consent_given = 1
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        enrolled = []
        for r in rows:
            try:
                emb_bytes = r['embedding_vector']
                emb_arr = np.frombuffer(emb_bytes, dtype=np.float32)
                enrolled.append((
                    r['student_id'],
                    r['full_name'],
                    r['roll_number'],
                    emb_arr
                ))
            except Exception as e:
                logger.error(f"Error parsing embedding for {r['student_id']}: {e}")

        return enrolled

    def enroll_student(self, data, image_np):
        """
        Enrolls a new student with face embedding and voluntary consent.
        data dict contains: student_id, full_name, roll_number, department, semester, email, consent_given
        image_np: OpenCV BGR image of student face
        """
        student_id = data.get('student_id', '').strip()
        full_name = data.get('full_name', '').strip()
        roll_number = data.get('roll_number', '').strip()
        department = data.get('department', '').strip()
        semester = data.get('semester', '').strip()
        email = data.get('email', '').strip()
        consent_given = 1 if data.get('consent_given') in [True, 1, '1', 'on'] else 0
        is_demo = 1 if data.get('is_demo_data') in [True, 1, '1', 'on'] else 0

        # Form validations
        if not student_id or not full_name or not roll_number:
            return False, "Student ID, Full Name, and Roll Number are required."

        if not consent_given:
            return False, "Voluntary consent is required to enroll biometric facial data."

        if image_np is None:
            return False, "A clear student photograph is required."

        # Detect face and extract embedding
        faces = self.face_service.detect_faces(image_np)
        if len(faces) == 0:
            return False, "No face detected in the image. Please ensure good lighting and face the camera directly."
        if len(faces) > 1:
            return False, f"Multiple faces detected ({len(faces)}). Please ensure only the student is in frame."

        face_meta = faces[0]
        embedding = self.face_service.extract_embedding(image_np, face_meta['raw_face'])
        if embedding is None:
            return False, "Failed to extract facial embedding features. Please capture a clearer photo."

        # Database duplicate checks
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM students WHERE student_id = ?", (student_id,))
        if cursor.fetchone():
            conn.close()
            return False, f"Student ID '{student_id}' is already registered."

        cursor.execute("SELECT id FROM students WHERE roll_number = ?", (roll_number,))
        if cursor.fetchone():
            conn.close()
            return False, f"Roll Number '{roll_number}' is already registered."

        try:
            # Save photo securely outside static directory
            safe_filename = f"{student_id}_{int(datetime.datetime.now().timestamp())}.jpg"
            photo_path = os.path.join(Config.PHOTOS_DIR, safe_filename)
            cv2.imwrite(photo_path, image_np)

            # Save embedding to binary blob and .npy file
            emb_filename = f"{student_id}_embedding.npy"
            emb_file_path = os.path.join(Config.EMBEDDINGS_DIR, emb_filename)
            np.save(emb_file_path, embedding)
            emb_blob = embedding.tobytes()

            # Insert student record
            cursor.execute("""
                INSERT INTO students (
                    student_id, full_name, roll_number, department, semester, email,
                    face_embedding_reference, photo_path, consent_given, is_demo_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                student_id, full_name, roll_number, department, semester, email,
                emb_filename, photo_path, consent_given, is_demo
            ))

            # Insert face embedding record
            cursor.execute("""
                INSERT INTO face_embeddings (student_id, embedding_vector, vector_dim)
                VALUES (?, ?, ?)
            """, (student_id, emb_blob, len(embedding)))

            conn.commit()
            conn.close()
            return True, f"Student {full_name} ({roll_number}) enrolled successfully!"
        except Exception as e:
            conn.rollback()
            conn.close()
            logger.error(f"Error enrolling student: {e}")
            return False, f"Database error during enrollment: {str(e)}"

    def delete_student(self, student_id):
        """Deletes a student and cascades deletion to files and embeddings."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT photo_path, face_embedding_reference FROM students WHERE student_id = ?", (student_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False, "Student not found."

        photo_path = row['photo_path']
        emb_ref = row['face_embedding_reference']

        try:
            # Delete physical files
            if photo_path and os.path.exists(photo_path):
                os.remove(photo_path)
            if emb_ref:
                npy_path = os.path.join(Config.EMBEDDINGS_DIR, emb_ref)
                if os.path.exists(npy_path):
                    os.remove(npy_path)

            # Delete DB records
            cursor.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
            cursor.execute("DELETE FROM face_embeddings WHERE student_id = ?", (student_id,))
            cursor.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
            conn.commit()
            conn.close()
            return True, f"Student {student_id} and associated biometric data deleted successfully."
        except Exception as e:
            conn.rollback()
            conn.close()
            logger.error(f"Error deleting student {student_id}: {e}")
            return False, f"Failed to delete student: {str(e)}"

    def mark_attendance(self, student_id, recognition_status='Present (Face Verified)', confidence=None, attendance_date=None, attendance_time=None):
        """
        Marks attendance for a student.
        Prevents duplicate entries on the same date.
        """
        now = datetime.datetime.now()
        date_str = attendance_date if attendance_date else now.strftime('%Y-%m-%d')
        time_str = attendance_time if attendance_time else now.strftime('%I:%M:%S %p')

        conn = get_db_connection()
        cursor = conn.cursor()

        # Check student exists
        cursor.execute("SELECT full_name, roll_number, department FROM students WHERE student_id = ?", (student_id,))
        student = cursor.fetchone()
        if not student:
            conn.close()
            return {
                'success': False,
                'already_marked': False,
                'message': f"Student ID '{student_id}' does not exist."
            }

        # Check if already marked for today
        cursor.execute("SELECT attendance_time, recognition_status FROM attendance WHERE student_id = ? AND attendance_date = ?", (student_id, date_str))
        existing = cursor.fetchone()
        if existing:
            conn.close()
            return {
                'success': True,
                'already_marked': True,
                'student_id': student_id,
                'full_name': student['full_name'],
                'roll_number': student['roll_number'],
                'department': student['department'],
                'attendance_date': date_str,
                'attendance_time': existing['attendance_time'],
                'status': existing['recognition_status'],
                'message': f"Attendance already marked for {student['full_name']} ({student['roll_number']}) today at {existing['attendance_time']}."
            }

        try:
            cursor.execute("""
                INSERT INTO attendance (student_id, attendance_date, attendance_time, recognition_status, confidence_score)
                VALUES (?, ?, ?, ?, ?)
            """, (student_id, date_str, time_str, recognition_status, confidence))
            conn.commit()
            conn.close()
            return {
                'success': True,
                'already_marked': False,
                'student_id': student_id,
                'full_name': student['full_name'],
                'roll_number': student['roll_number'],
                'department': student['department'],
                'attendance_date': date_str,
                'attendance_time': time_str,
                'status': recognition_status,
                'confidence': confidence,
                'message': f"Attendance marked: {student['full_name']} ({student['roll_number']}) at {time_str}"
            }
        except sqlite3.IntegrityError:
            conn.close()
            return {
                'success': True,
                'already_marked': True,
                'student_id': student_id,
                'full_name': student['full_name'],
                'roll_number': student['roll_number'],
                'message': f"Attendance already recorded today for {student['full_name']}."
            }
        except Exception as e:
            conn.rollback()
            conn.close()
            logger.error(f"Error marking attendance: {e}")
            return {
                'success': False,
                'already_marked': False,
                'message': f"Database error: {str(e)}"
            }

    def process_camera_frame(self, frame_b64, custom_threshold=None):
        """
        Processes a live video frame from the client webcam:
        1. Decodes base64 frame.
        2. Detects faces.
        3. Matches against enrolled students.
        4. If match meets threshold, marks attendance.
        5. Returns structured JSON with boxes, match status, student info, and friendly UI messages.
        """
        img = self.face_service.decode_base64_image(frame_b64)
        if img is None:
            return {
                'success': False,
                'detected': False,
                'message': 'Failed to decode camera frame.'
            }

        faces = self.face_service.detect_faces(img)
        if not faces:
            return {
                'success': True,
                'detected': False,
                'message': 'Looking for face...',
                'faces': []
            }

        # Determine threshold
        if custom_threshold is not None:
            threshold = float(custom_threshold)
        else:
            thresh_setting = get_setting('similarity_threshold', Config.DEFAULT_SIMILARITY_THRESHOLD)
            threshold = float(thresh_setting)

        enrolled = self.get_enrolled_students_with_embeddings()
        processed_faces = []
        recognized_any = False
        primary_result = None

        for face in faces:
            box = face['box']
            raw_face = face['raw_face']
            emb = self.face_service.extract_embedding(img, raw_face)
            
            match_student, score, score_pct = self.face_service.find_best_match(emb, enrolled, threshold=threshold)

            face_res = {
                'box': box,
                'detected_confidence': round(face['confidence'], 2),
                'score': round(float(score), 3),
                'score_pct': round(score_pct, 1)
            }

            if match_student:
                recognized_any = True
                face_res['recognized'] = True
                face_res['student_id'] = match_student['student_id']
                face_res['full_name'] = match_student['full_name']
                face_res['roll_number'] = match_student['roll_number']

                # Mark attendance
                att_res = self.mark_attendance(
                    student_id=match_student['student_id'],
                    recognition_status='Present (Face Verified)',
                    confidence=score
                )
                face_res['attendance_result'] = att_res

                if not primary_result:
                    primary_result = {
                        'student': match_student,
                        'attendance': att_res,
                        'score_pct': round(score_pct, 1)
                    }
            else:
                face_res['recognized'] = False
                face_res['full_name'] = 'Unknown Person'
                face_res['roll_number'] = 'N/A'

            processed_faces.append(face_res)

        return {
            'success': True,
            'detected': True,
            'faces_count': len(faces),
            'recognized': recognized_any,
            'primary_result': primary_result,
            'faces': processed_faces,
            'threshold_used': threshold
        }

    def get_dashboard_statistics(self, target_date=None):
        """
        Calculates key dashboard summary cards and chart metrics.
        - Total Registered Students
        - Students Present Today
        - Students Absent Today
        - Today's Attendance Percentage
        """
        if not target_date:
            target_date = datetime.datetime.now().strftime('%Y-%m-%d')

        conn = get_db_connection()
        cursor = conn.cursor()

        # 1. Total Registered Students
        cursor.execute("SELECT COUNT(*) AS total FROM students")
        total_students = cursor.fetchone()['total']

        # 2. Present Today
        cursor.execute("SELECT COUNT(DISTINCT student_id) AS present FROM attendance WHERE attendance_date = ?", (target_date,))
        present_today = cursor.fetchone()['present']

        # 3. Absent Today
        absent_today = max(0, total_students - present_today)

        # 4. Percentage
        attendance_percentage = round((present_today / total_students * 100), 1) if total_students > 0 else 0.0

        # 5. Recent Attendance Logs (latest 10 entries)
        cursor.execute("""
            SELECT a.id, a.student_id, a.attendance_date, a.attendance_time, a.recognition_status, a.confidence_score,
                   s.full_name, s.roll_number, s.department, s.semester
            FROM attendance a
            JOIN students s ON a.student_id = s.student_id
            WHERE a.attendance_date = ?
            ORDER BY a.id DESC LIMIT 10
        """, (target_date,))
        recent_logs = [dict(r) for r in cursor.fetchall()]

        # 6. Department breakdown for today
        cursor.execute("""
            SELECT s.department, 
                   COUNT(DISTINCT s.student_id) AS total_enrolled,
                   COUNT(DISTINCT a.student_id) AS present_count
            FROM students s
            LEFT JOIN attendance a ON s.student_id = a.student_id AND a.attendance_date = ?
            GROUP BY s.department
        """, (target_date,))
        dept_data = [dict(r) for r in cursor.fetchall()]

        # 7. Last 7 days attendance trend
        cursor.execute("""
            SELECT attendance_date, COUNT(DISTINCT student_id) AS count
            FROM attendance
            GROUP BY attendance_date
            ORDER BY attendance_date DESC LIMIT 7
        """)
        trend_rows = cursor.fetchall()
        trend_data = [{'date': r['attendance_date'], 'count': r['count']} for r in reversed(trend_rows)]

        conn.close()

        return {
            'target_date': target_date,
            'total_students': total_students,
            'present_today': present_today,
            'absent_today': absent_today,
            'attendance_percentage': attendance_percentage,
            'recent_logs': recent_logs,
            'dept_data': dept_data,
            'trend_data': trend_data
        }

    def get_attendance_records(self, filters=None):
        """
        Retrieves attendance records with optional filtering:
        date, student_name, roll_number, department, semester.
        """
        filters = filters or {}
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
            SELECT a.id, a.student_id, a.attendance_date, a.attendance_time, a.recognition_status, a.confidence_score,
                   s.full_name, s.roll_number, s.department, s.semester, s.email, a.is_demo_record
            FROM attendance a
            JOIN students s ON a.student_id = s.student_id
            WHERE 1=1
        """
        params = []

        if filters.get('date'):
            query += " AND a.attendance_date = ?"
            params.append(filters['date'])

        if filters.get('department'):
            query += " AND s.department = ?"
            params.append(filters['department'])

        if filters.get('semester'):
            query += " AND s.semester = ?"
            params.append(filters['semester'])

        if filters.get('search'):
            term = f"%{filters['search'].strip()}%"
            query += " AND (s.full_name LIKE ? OR s.roll_number LIKE ? OR s.student_id LIKE ?)"
            params.extend([term, term, term])

        query += " ORDER BY a.attendance_date DESC, a.attendance_time DESC"
        cursor.execute(query, params)
        records = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return records

    def get_all_students_with_stats(self):
        """
        Returns all registered students along with their total attendance count and percentage.
        """
        conn = get_db_connection()
        cursor = conn.cursor()

        # Total distinct dates where attendance was taken
        cursor.execute("SELECT COUNT(DISTINCT attendance_date) AS total_days FROM attendance")
        total_session_days = cursor.fetchone()['total_days']
        total_session_days = max(1, total_session_days)

        query = """
            SELECT s.*, 
                   COUNT(a.id) AS days_present,
                   MAX(a.attendance_date) AS last_present_date
            FROM students s
            LEFT JOIN attendance a ON s.student_id = a.student_id
            GROUP BY s.student_id
            ORDER BY s.roll_number ASC
        """
        cursor.execute(query)
        students = []
        for r in cursor.fetchall():
            d = dict(r)
            d['percentage'] = round((d['days_present'] / total_session_days * 100), 1)
            students.append(d)

        conn.close()
        return students, total_session_days

    def export_attendance_csv(self, filters=None):
        """Generates a CSV string using Pandas for filtered attendance records."""
        records = self.get_attendance_records(filters)
        if not records:
            df = pd.DataFrame(columns=[
                'Attendance ID', 'Student ID', 'Roll Number', 'Full Name',
                'Department', 'Semester', 'Date', 'Time', 'Status', 'Confidence Score'
            ])
        else:
            data = []
            for r in records:
                conf = f"{round(r['confidence_score'] * 100, 1)}%" if r['confidence_score'] is not None else "N/A"
                data.append({
                    'Attendance ID': r['id'],
                    'Student ID': r['student_id'],
                    'Roll Number': r['roll_number'],
                    'Full Name': r['full_name'],
                    'Department': r['department'],
                    'Semester': r['semester'],
                    'Date': r['attendance_date'],
                    'Time': r['attendance_time'],
                    'Status': r['recognition_status'],
                    'Confidence Score': conf
                })
            df = pd.DataFrame(data)

        output = io.StringIO()
        df.to_csv(output, index=False)
        return output.getvalue()

    def export_daily_register_csv(self, target_date=None):
        """
        Generates a full daily register CSV showing all students with Present / Absent status
        (matching Table 2 of the workshop report).
        """
        if not target_date:
            target_date = datetime.datetime.now().strftime('%Y-%m-%d')

        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
            SELECT s.student_id, s.roll_number, s.full_name, s.department, s.semester,
                   a.attendance_time, a.recognition_status
            FROM students s
            LEFT JOIN attendance a ON s.student_id = a.student_id AND a.attendance_date = ?
            ORDER BY s.roll_number ASC
        """
        cursor.execute(query, (target_date,))
        rows = cursor.fetchall()
        conn.close()

        data = []
        for r in rows:
            is_present = bool(r['recognition_status'])
            data.append({
                'Student ID': r['student_id'],
                'Roll No': r['roll_number'],
                'Student Name': r['full_name'],
                'Department': r['department'],
                'Semester': r['semester'],
                'Date': target_date,
                'Time': r['attendance_time'] if is_present else '-',
                'Status': 'Present' if is_present else 'Absent'
            })

        df = pd.DataFrame(data)
        output = io.StringIO()
        df.to_csv(output, index=False)
        return output.getvalue()
