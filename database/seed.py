import os
import sys
import datetime
import numpy as np
import cv2

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from database.db import get_db_connection, init_db

# 10 Fictional Students from the Workshop Report (Page 11)
DEMO_STUDENTS = [
    {
        'student_id': 'ST001',
        'full_name': 'Aarav Kumar',
        'roll_number': 'CSE001',
        'department': 'Computer Science & Engineering',
        'semester': '5th',
        'email': 'aarav.cse001@polytechnic.ac.in',
        'sample_time': '09:02:15 AM',
        'sample_present': True
    },
    {
        'student_id': 'ST002',
        'full_name': 'Ananya Sharma',
        'roll_number': 'CSE002',
        'department': 'Computer Science & Engineering',
        'semester': '5th',
        'email': 'ananya.cse002@polytechnic.ac.in',
        'sample_time': '09:03:40 AM',
        'sample_present': True
    },
    {
        'student_id': 'ST003',
        'full_name': 'Riya Singh',
        'roll_number': 'CSE003',
        'department': 'Computer Science & Engineering',
        'semester': '5th',
        'email': 'riya.cse003@polytechnic.ac.in',
        'sample_time': None,
        'sample_present': False  # Absent in Table 2
    },
    {
        'student_id': 'ST004',
        'full_name': 'Aditya Verma',
        'roll_number': 'CSE004',
        'department': 'Computer Science & Engineering',
        'semester': '5th',
        'email': 'aditya.cse004@polytechnic.ac.in',
        'sample_time': '09:04:18 AM',
        'sample_present': True
    },
    {
        'student_id': 'ST005',
        'full_name': 'Priya Kumari',
        'roll_number': 'CSE005',
        'department': 'Computer Science & Engineering',
        'semester': '5th',
        'email': 'priya.cse005@polytechnic.ac.in',
        'sample_time': '09:05:52 AM',
        'sample_present': True
    },
    {
        'student_id': 'ST006',
        'full_name': 'Rahul Kumar',
        'roll_number': 'CSE006',
        'department': 'Computer Science & Engineering',
        'semester': '5th',
        'email': 'rahul.cse006@polytechnic.ac.in',
        'sample_time': None,
        'sample_present': False  # Absent in Table 2
    },
    {
        'student_id': 'ST007',
        'full_name': 'Sneha Gupta',
        'roll_number': 'CSE007',
        'department': 'Computer Science & Engineering',
        'semester': '5th',
        'email': 'sneha.cse007@polytechnic.ac.in',
        'sample_time': '09:06:33 AM',
        'sample_present': True
    },
    {
        'student_id': 'ST008',
        'full_name': 'Aman Raj',
        'roll_number': 'CSE008',
        'department': 'Computer Science & Engineering',
        'semester': '5th',
        'email': 'aman.cse008@polytechnic.ac.in',
        'sample_time': '09:07:11 AM',
        'sample_present': True
    },
    {
        'student_id': 'ST009',
        'full_name': 'Neha Kumari',
        'roll_number': 'CSE009',
        'department': 'Computer Science & Engineering',
        'semester': '5th',
        'email': 'neha.cse009@polytechnic.ac.in',
        'sample_time': '09:08:45 AM',
        'sample_present': True
    },
    {
        'student_id': 'ST010',
        'full_name': 'Rohit Singh',
        'roll_number': 'CSE010',
        'department': 'Computer Science & Engineering',
        'semester': '5th',
        'email': 'rohit.cse010@polytechnic.ac.in',
        'sample_time': None,
        'sample_present': False  # Absent in Table 2
    }
]

def create_synthetic_avatar(name, roll_no, path):
    """Creates a clean colored avatar image with initials for demonstration purposes."""
    img = np.ones((250, 250, 3), dtype=np.uint8) * 240
    # Gradient/Color based on hash of name
    h_val = abs(hash(name)) % 360
    color = (int(50 + (h_val % 100)), int(100 + (h_val % 80)), int(180 + (h_val % 70)))
    
    # Outer circle
    cv2.circle(img, (125, 125), 100, color, -1)
    
    # Initials
    initials = "".join([part[0] for part in name.split()[:2]]).upper()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.2
    thickness = 3
    text_size = cv2.getTextSize(initials, font, font_scale, thickness)[0]
    text_x = (250 - text_size[0]) // 2
    text_y = (250 + text_size[1]) // 2 - 10
    cv2.putText(img, initials, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    # Subtitle
    cv2.putText(img, roll_no, (70, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1, cv2.LINE_AA)
    
    cv2.imwrite(path, img)

def seed_database(include_demo_attendance=True):
    """Populates the database with 10 sample students and optional demo attendance records."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    print("[*] Starting database seed...")
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    report_sample_date = "2026-10-09"  # Workshop sample date from report

    for idx, s in enumerate(DEMO_STUDENTS, start=1):
        # 1. Insert or update student
        cursor.execute("SELECT id FROM students WHERE student_id = ?", (s['student_id'],))
        existing = cursor.fetchone()

        photo_filename = f"{s['student_id']}_avatar.jpg"
        photo_path = os.path.join(Config.PHOTOS_DIR, photo_filename)
        create_synthetic_avatar(s['full_name'], s['roll_number'], photo_path)

        # Generate a distinct deterministic 128D unit vector
        np.random.seed(1000 + idx)
        raw_vec = np.random.randn(128).astype(np.float32)
        unit_vec = raw_vec / np.linalg.norm(raw_vec)
        emb_filename = f"{s['student_id']}_embedding.npy"
        emb_path = os.path.join(Config.EMBEDDINGS_DIR, emb_filename)
        np.save(emb_path, unit_vec)
        emb_blob = unit_vec.tobytes()

        if not existing:
            cursor.execute("""
                INSERT INTO students (
                    student_id, full_name, roll_number, department, semester, email,
                    face_embedding_reference, photo_path, consent_given, is_demo_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 1)
            """, (
                s['student_id'], s['full_name'], s['roll_number'], s['department'], s['semester'],
                s['email'], emb_filename, photo_path
            ))
            cursor.execute("""
                INSERT INTO face_embeddings (student_id, embedding_vector, vector_dim)
                VALUES (?, ?, 128)
            """, (s['student_id'], emb_blob))
            print(f"  [+] Enrolled student: {s['full_name']} ({s['roll_number']})")
        else:
            print(f"  [.] Student already present: {s['full_name']} ({s['roll_number']})")

        # 2. Insert clearly labeled demonstration attendance records if requested
        if include_demo_attendance and s['sample_present']:
            for att_date in [today_str, report_sample_date]:
                cursor.execute(
                    "SELECT id FROM attendance WHERE student_id = ? AND attendance_date = ?",
                    (s['student_id'], att_date)
                )
                if not cursor.fetchone():
                    cursor.execute("""
                        INSERT INTO attendance (
                            student_id, attendance_date, attendance_time,
                            recognition_status, confidence_score, is_demo_record
                        ) VALUES (?, ?, ?, ?, ?, 1)
                    """, (
                        s['student_id'],
                        att_date,
                        s['sample_time'],
                        'Present (Demo Sample)',
                        0.94
                    ))

    conn.commit()
    conn.close()
    print("[+] Database seeding complete! 10 sample students ready.")

if __name__ == '__main__':
    seed_database(include_demo_attendance=True)
