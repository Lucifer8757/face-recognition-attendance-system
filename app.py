import os
import io
import datetime
from functools import wraps
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, jsonify, Response, send_file
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
import cv2
import numpy as np

from config import Config
from database import get_db_connection, init_db, get_setting, update_setting
from database.seed import seed_database
from services.attendance_service import AttendanceService
from services.face_recognition_service import FaceRecognitionService

app = Flask(__name__)
app.config.from_object(Config)

# Initialize services
attendance_service = AttendanceService()
face_service = FaceRecognitionService()

# Ensure database and storage exist
init_db()

def login_required(f):
    """Decorator to enforce admin authentication on protected routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in with admin credentials to access this section.', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# -------------------------------------------------------------
# AUTHENTICATION ROUTES
# -------------------------------------------------------------

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM admin_users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            flash(f'Welcome back, {user["full_name"] or user["username"]}!', 'success')
            next_url = request.args.get('next')
            return redirect(next_url or url_for('dashboard'))
        else:
            flash('Invalid admin username or password. Please try again.', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out securely.', 'info')
    return redirect(url_for('login'))

# -------------------------------------------------------------
# CORE DASHBOARD
# -------------------------------------------------------------

@app.route('/')
@app.route('/dashboard')
@login_required
def dashboard():
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    stats = attendance_service.get_dashboard_statistics(today_str)
    return render_template('dashboard.html', stats=stats, active_page='dashboard')

# -------------------------------------------------------------
# STUDENT REGISTRATION MODULE
# -------------------------------------------------------------

@app.route('/register', methods=['GET', 'POST'])
@login_required
def register_student():
    if request.method == 'POST':
        data = {
            'student_id': request.form.get('student_id', '').strip(),
            'full_name': request.form.get('full_name', '').strip(),
            'roll_number': request.form.get('roll_number', '').strip(),
            'department': request.form.get('department', '').strip(),
            'semester': request.form.get('semester', '').strip(),
            'email': request.form.get('email', '').strip(),
            'consent_given': request.form.get('consent_given')
        }

        image_np = None

        # Check if webcam snapshot was submitted
        captured_b64 = request.form.get('captured_image_b64')
        if captured_b64 and len(captured_b64) > 100:
            image_np = face_service.decode_base64_image(captured_b64)

        # Check if a file was uploaded
        elif 'photo_file' in request.files and request.files['photo_file'].filename:
            file = request.files['photo_file']
            ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
            if ext not in Config.ALLOWED_EXTENSIONS:
                flash('Invalid file format. Allowed types: JPG, PNG, WEBP.', 'danger')
                return render_template('register_student.html', active_page='register')
            
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_np is None:
            flash('Please take a webcam snapshot or upload a valid photo.', 'danger')
            return render_template('register_student.html', active_page='register')

        success, message = attendance_service.enroll_student(data, image_np)
        if success:
            flash(message, 'success')
            return redirect(url_for('face_attendance'))
        else:
            flash(message, 'danger')

    return render_template('register_student.html', active_page='register')

# -------------------------------------------------------------
# LIVE FACE RECOGNITION ATTENDANCE MODULE
# -------------------------------------------------------------

@app.route('/attendance')
@login_required
def face_attendance():
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    current_thresh = get_setting('similarity_threshold', Config.DEFAULT_SIMILARITY_THRESHOLD)
    today_records = attendance_service.get_attendance_records({'date': today_str})
    return render_template(
        'face_attendance.html',
        current_threshold=current_thresh,
        today_records=today_records,
        active_page='face_attendance'
    )

@app.route('/api/recognize_frame', methods=['POST'])
@login_required
def api_recognize_frame():
    """API endpoint receiving client video frame for detection and matching."""
    data = request.get_json() or {}
    frame_b64 = data.get('frame')
    threshold = data.get('threshold')

    if not frame_b64:
        return jsonify({'success': False, 'message': 'No frame received.'}), 400

    result = attendance_service.process_camera_frame(frame_b64, custom_threshold=threshold)
    return jsonify(result)

# -------------------------------------------------------------
# ATTENDANCE RECORDS & MANAGEMENT
# -------------------------------------------------------------

@app.route('/records')
@login_required
def attendance_records():
    search = request.args.get('search', '').strip()
    date = request.args.get('date', '').strip()
    department = request.args.get('department', '').strip()
    semester = request.args.get('semester', '').strip()

    filters = {
        'search': search,
        'date': date,
        'department': department,
        'semester': semester
    }

    records = attendance_service.get_attendance_records(filters)
    
    # Get all students for manual entry modal dropdown
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT student_id, roll_number, full_name, department FROM students ORDER BY roll_number ASC")
    all_students = cursor.fetchall()
    conn.close()

    today_date = datetime.date.today().strftime('%Y-%m-%d')

    return render_template(
        'attendance_records.html',
        records=records,
        all_students=all_students,
        current_filters=filters,
        today_date=today_date,
        active_page='records'
    )

@app.route('/manual_mark', methods=['POST'])
@login_required
def manual_mark_attendance():
    """Manual attendance entry/correction for authorized administrator."""
    student_id = request.form.get('student_id')
    att_date = request.form.get('attendance_date')
    att_time_raw = request.form.get('attendance_time', '09:00')
    status = request.form.get('recognition_status', 'Present (Manual Admin)')

    # Format time
    try:
        dt = datetime.datetime.strptime(att_time_raw, '%H:%M')
        time_formatted = dt.strftime('%I:%M %p')
    except Exception:
        time_formatted = att_time_raw

    res = attendance_service.mark_attendance(
        student_id=student_id,
        recognition_status=status,
        confidence=1.0,
        attendance_date=att_date,
        attendance_time=time_formatted
    )

    if res['success']:
        if res.get('already_marked'):
            flash(res['message'], 'warning')
        else:
            flash(f"Manual attendance marked for {res['full_name']} ({res['roll_number']}).", 'success')
    else:
        flash(res['message'], 'danger')

    return redirect(url_for('attendance_records'))

@app.route('/delete_attendance/<int:record_id>', methods=['POST'])
@login_required
def delete_attendance_record(record_id):
    """Deletes an attendance entry (admin correction)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()
    flash(f"Attendance record #{record_id} removed.", 'info')
    return redirect(request.referrer or url_for('attendance_records'))

# -------------------------------------------------------------
# REPORTS & ANALYTICS
# -------------------------------------------------------------

@app.route('/reports')
@login_required
def reports():
    students, total_days = attendance_service.get_all_students_with_stats()
    
    # Calculate institution-wide statistics
    if students:
        avg_att = round(sum(s['percentage'] for s in students) / len(students), 1)
        eligible = sum(1 for s in students if s['percentage'] >= 75.0)
    else:
        avg_att = 0.0
        eligible = 0

    return render_template(
        'reports.html',
        students=students,
        total_session_days=total_days,
        avg_attendance=avg_att,
        eligible_count=eligible,
        active_page='reports'
    )

@app.route('/delete_student/<student_id>', methods=['POST'])
@login_required
def delete_student_profile(student_id):
    """Right to erasure: permanently purges student, face embeddings and photos."""
    success, msg = attendance_service.delete_student(student_id)
    if success:
        flash(msg, 'success')
    else:
        flash(msg, 'danger')
    return redirect(url_for('reports'))

# -------------------------------------------------------------
# CSV EXPORTS
# -------------------------------------------------------------

@app.route('/export/attendance_csv')
@login_required
def export_attendance_csv():
    filters = {
        'search': request.args.get('search'),
        'date': request.args.get('date'),
        'department': request.args.get('department'),
        'semester': request.args.get('semester')
    }
    csv_str = attendance_service.export_attendance_csv(filters)
    filename = f"attendance_report_{datetime.date.today().strftime('%Y%m%d')}.csv"
    return Response(
        csv_str,
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename={filename}"}
    )

@app.route('/export/daily_sheet')
@login_required
def export_daily_sheet():
    target_date = request.args.get('date', datetime.date.today().strftime('%Y-%m-%d'))
    csv_str = attendance_service.export_daily_register_csv(target_date)
    filename = f"daily_attendance_sheet_{target_date.replace('-', '')}.csv"
    return Response(
        csv_str,
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename={filename}"}
    )

# -------------------------------------------------------------
# SETTINGS & PRIVACY
# -------------------------------------------------------------

@app.route('/settings')
@login_required
def settings():
    settings_dict = {
        'similarity_threshold': get_setting('similarity_threshold', str(Config.DEFAULT_SIMILARITY_THRESHOLD)),
        'liveness_detection': get_setting('liveness_detection', '0'),
        'institution_name': get_setting('institution_name', "Government Women's Polytechnic, Ranchi"),
        'academic_year': get_setting('academic_year', '2026')
    }
    return render_template('settings.html', settings=settings_dict, active_page='settings')

@app.route('/settings/update', methods=['POST'])
@login_required
def update_settings():
    thresh = request.form.get('similarity_threshold', '0.38')
    liveness = '1' if request.form.get('liveness_detection') == '1' else '0'
    inst = request.form.get('institution_name', '').strip()
    year = request.form.get('academic_year', '').strip()

    update_setting('similarity_threshold', thresh)
    update_setting('liveness_detection', liveness)
    if inst:
        update_setting('institution_name', inst)
    if year:
        update_setting('academic_year', year)

    flash('System settings updated successfully.', 'success')
    return redirect(url_for('settings'))

@app.route('/reseed_database', methods=['POST'])
@login_required
def reseed_database_route():
    seed_database(include_demo_attendance=True)
    flash('Demonstration dataset re-seeded successfully with 10 sample students.', 'success')
    return redirect(url_for('dashboard'))

# -------------------------------------------------------------
# SECURE PHOTO SERVING (NON-PUBLIC)
# -------------------------------------------------------------

@app.route('/admin/student_photo/<student_id>')
@login_required
def secure_student_photo(student_id):
    """Protected route to serve student photographs only to authenticated admins."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT photo_path FROM students WHERE student_id = ?", (student_id,))
    row = cursor.fetchone()
    conn.close()

    if row and row['photo_path'] and os.path.exists(row['photo_path']):
        return send_file(row['photo_path'], mimetype='image/jpeg')
    return Response(status=404)

# -------------------------------------------------------------
# ERROR HANDLERS
# -------------------------------------------------------------

@app.errorhandler(404)
def not_found_error(error):
    return render_template('base.html', content="<div class='container py-5 text-center'><h3>Page Not Found</h3><p>The requested page does not exist.</p><a href='/dashboard' class='btn btn-primary'>Return to Dashboard</a></div>"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('base.html', content="<div class='container py-5 text-center'><h3>Internal Server Error</h3><p>An unexpected error occurred. Please try again.</p><a href='/dashboard' class='btn btn-primary'>Return to Dashboard</a></div>"), 500

if __name__ == '__main__':
    print("=" * 65)
    print("  AI-Based Face Recognition Attendance System")
    print("  Institution: Government Women's Polytechnic, Ranchi")
    print("  Server starting on: http://127.0.0.1:5000")
    print("  Default Admin Login: admin / admin123")
    print("=" * 65)
    app.run(host='127.0.0.1', port=5000, debug=True)
