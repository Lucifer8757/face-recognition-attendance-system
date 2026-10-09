# AI-Based Face Recognition Attendance System for Students

[![Python](https://img.shields.io/badge/Python-3.9%20to%203.14+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-YuNet%20%2B%20SFace-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white)](https://getbootstrap.com)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)

**Academic Workshop Project | Generation AI (2026)**  
- **Author:** Maully Singh (Diploma in Computer Science & Engineering, 5th Semester)  
- **Institution:** Government Women's Polytechnic, Ranchi  
- **Workshop:** Anudip Foundation — Machine Learning for Data Science Projects  

---

## 1. Project Overview & Motivation

Traditional roll calls and attendance registers consume 10 to 15 minutes of every lecture, are prone to proxy attendance, and require manual counting to calculate monthly attendance percentages.

This **AI-Based Face Recognition Attendance System** automates the entire attendance lifecycle:
1. **Detects** faces from a live webcam feed in real time.
2. **Extracts** a unique 128-dimensional mathematical vector (embedding) using OpenCV deep learning neural networks.
3. **Compares** detected vectors against registered student embeddings using cosine similarity.
4. **Marks** attendance automatically with the exact date and timestamp upon matching.
5. **Prevents** duplicate attendance entries for the same student on the same date via database constraints.
6. **Generates** automated daily and cumulative attendance reports with 1-click CSV export and Chart.js analytics.

---

## 2. System Architecture & 6-Step Workflow

As outlined in the workshop report, the system processes visual data through six sequential phases:

```
[ Step 1: Camera & Face Detection ]
       │  YuNet CNN detects bounding box & 5 facial landmarks (eyes, nose, mouth)
       ▼
[ Step 2: Preprocessing & Alignment ]
       │  Affine transformation rotates, centers, and crops face to 112×112 standard format
       ▼
[ Step 3: Feature Extraction ]
       │  SFace deep neural network extracts 128-dimensional normalized unit vector
       ▼
[ Step 4: Face Matching (Cosine Similarity) ]
       │  Computes dot product: Similarity = (u · v) / (||u|| * ||v||)
       ▼
[ Step 5: Identification & Decision Logic ]
       │  Score ≥ Threshold (default 0.38) ──► Match Verified (Student Identified)
       │  Score < Threshold                 ──► Unknown Person (Rejected)
       ▼
[ Step 6: Attendance Record & Constraint Check ]
       │  Checks UNIQUE(student_id, attendance_date) constraint
       │  If new: Saves Name, Roll No, Date, Time, Status ──► Green Toast & Chime
       │  If already present today: Notifies "Already Marked" ──► No Duplicate
```

---

## 3. Technology Stack & Modern Deep Learning Decisions

| Component | Technology | Rationale |
|---|---|---|
| **Frontend** | HTML5, CSS3, JavaScript (WebRTC, Canvas API, Web Audio API), Bootstrap 5.3 | Responsive academic UI, dark navy theme, live bounding-box HUD, browser camera access without driver lockups. |
| **Backend** | Python with Flask | Lightweight, fast request handling, clean REST API for JSON frame evaluation. |
| **Face Detection** | OpenCV **YuNet** (`cv2.FaceDetectorYN`) | Official OpenCV deep learning CNN detector (232 KB ONNX model). Extremely fast, detects faces at various angles, and outputs 5 landmarks. |
| **Face Recognition** | OpenCV **SFace** (`cv2.FaceRecognizerSF`) | Official OpenCV SphereFace/MobileFaceNet model (38 MB ONNX). Generates true 128D embeddings and cosine matching without external C++ compilers. |
| **Database** | SQLite3 | Relational database with foreign keys, indexes, and a `UNIQUE(student_id, attendance_date)` constraint. |
| **Data Processing** | Pandas & NumPy | Vector normalization, cosine calculations, and instant CSV report generation. |
| **Visual Analytics** | Chart.js 4.4 | Real-time doughnut chart (Present vs Absent) and attendance trend lines. |

### Technical Note: Why OpenCV SFace Instead of `face_recognition` (dlib)?
In Python 3.12+ and on Windows machines, attempting to install `face_recognition` frequently crashes during pip install with:
```
CMake Error at CMakeLists.txt:5: You must use Visual Studio to build a python extension on windows.
ERROR: Failed building wheel for dlib
```
To eliminate this barrier for students and workshop attendees, this project uses **OpenCV's native SFace engine**. It provides identical 128D biometric feature vectors, matches or exceeds dlib's accuracy, and runs natively inside pre-compiled `opencv-python` wheels with **zero CMake or Visual C++ compilation requirements**.

---

## 4. Database Design

The SQLite database (`data/face_attendance.db`) is structured into modular tables with strict relational integrity:

```
┌───────────────────────────────┐          ┌───────────────────────────────────┐
│           STUDENTS            │          │             ATTENDANCE            │
├───────────────────────────────┤          ├───────────────────────────────────┤
│ id (PK)                       │ 1      N │ id (PK)                           │
│ student_id (TEXT, UNIQUE) ────┼──────────┼─► student_id (FK)                 │
│ full_name (TEXT)              │  marks   │ attendance_date (TEXT, YYYY-MM-DD)│
│ roll_number (TEXT, UNIQUE)    │          │ attendance_time (TEXT, HH:MM AM)  │
│ department (TEXT)             │          │ recognition_status (TEXT)         │
│ semester (TEXT)               │          │ confidence_score (REAL)           │
│ email (TEXT)                  │          │ is_demo_record (INT)              │
│ face_embedding_reference      │          │ created_at (TIMESTAMP)            │
│ photo_path (TEXT)             │          ├───────────────────────────────────┤
│ consent_given (INT)           │          │ UNIQUE(student_id,attendance_date)│
│ created_at (TIMESTAMP)        │          └───────────────────────────────────┘
└──────────────┬────────────────┘
               │ 1
               │
               │ 1
┌──────────────▼────────────────┐
│        FACE_EMBEDDINGS        │
├───────────────────────────────┤
│ id (PK)                       │
│ student_id (FK) ──────────────┘
│ embedding_vector (BLOB, 128D) │
│ vector_dim (INT, 128)         │
│ created_at (TIMESTAMP)        │
└───────────────────────────────┘
```

> **Biometric Privacy Design:** Face embeddings and photographs are stored in `data/secure_storage/` outside the web root (`static/`), preventing unauthorized direct HTTP downloads.

---

## 5. Sample Dataset (Pre-Seeded 10 Students)

The system includes a seed script that populates the 10 fictional students from **Table 1 & Table 2 of the workshop report**:

| Student ID | Student Name | Roll Number | Department | Semester | 09/10/2026 Sample Status |
|---|---|---|---|---|---|
| `ST001` | Aarav Kumar | CSE001 | Computer Science & Eng. | 5th | **Present** (09:02 AM) |
| `ST002` | Ananya Sharma | CSE002 | Computer Science & Eng. | 5th | **Present** (09:03 AM) |
| `ST003` | Riya Singh | CSE003 | Computer Science & Eng. | 5th | **Absent** |
| `ST004` | Aditya Verma | CSE004 | Computer Science & Eng. | 5th | **Present** (09:04 AM) |
| `ST005` | Priya Kumari | CSE005 | Computer Science & Eng. | 5th | **Present** (09:05 AM) |
| `ST006` | Rahul Kumar | CSE006 | Computer Science & Eng. | 5th | **Absent** |
| `ST007` | Sneha Gupta | CSE007 | Computer Science & Eng. | 5th | **Present** (09:06 AM) |
| `ST008` | Aman Raj | CSE008 | Computer Science & Eng. | 5th | **Present** (09:07 AM) |
| `ST009` | Neha Kumari | CSE009 | Computer Science & Eng. | 5th | **Present** (09:08 AM) |
| `ST010` | Rohit Singh | CSE010 | Computer Science & Eng. | 5th | **Absent** |

**Daily Summary Result:** 7 Present, 3 Absent = **70% Turnout** (exact match with workshop report Page 11).

---

## 6. Step-by-Step Installation & Local Run

### Prerequisites
- Python 3.9, 3.10, 3.11, 3.12, 3.13, or 3.14.
- A functional laptop webcam or external USB camera.
- A modern web browser (Google Chrome, Microsoft Edge, Mozilla Firefox).

### Step 1: Open Terminal in Project Directory
```powershell
cd C:\Users\ankit\.gemini\antigravity\scratch\face-attendance-system
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Seed Database with Sample Students
```powershell
python database/seed.py
```
*This initializes the SQLite database schema, creates the admin user, and populates the 10 fictional students.*

### Step 4: Run the Flask Web Application
```powershell
python app.py
```

### Step 5: Access the Web Portal
Open your web browser and navigate to:
**`http://127.0.0.1:5000`**

- **Username:** `admin`
- **Password:** `admin123`

---

## 7. How to Test Face Recognition with Your Webcam (Live Demonstration)

Follow these steps for a live presentation or workshop demo:

### Step A: Enroll Your Own Face as Presenter
1. Log in and navigate to **Student Registration** in the sidebar.
2. Click the top button: **"Pre-fill Presenter Details"** (populates Maully Singh / Workshop Presenter).
3. Under the *Facial Biometric Photograph* section, click **"Start Camera"**.
4. Align your face inside the blue oval guide and click **"Snap Photo"**.
5. Ensure the **Informed Biometric Consent** checkbox is checked.
6. Click **"Enroll Student & Extract Biometrics"**.  
   *The system extracts your 128D facial embedding, stores it securely, and redirects to the scanner.*

### Step B: Test Live Face Recognition
1. Navigate to **Face Recognition** in the sidebar.
2. Click **"Start Camera"** and allow browser camera permissions if prompted.
3. Look directly into the webcam:
   - Within 1 second, a **green bounding box** will lock onto your face.
   - The HUD banner will display: `[Your Name] ([Confidence %])`.
   - A pleasant double-chime will sound.
   - A green toast will confirm: **"Attendance Marked!"** with the exact timestamp.
   - Your name will instantly appear at the top of the **Today's Attendance Feed** on the right side.

### Step C: Test Duplicate Attendance Prevention
1. Keep looking into the webcam or step away and return.
2. The system detects you again, but recognizes you are already marked:
   - The status bar displays in blue: **"Attendance already marked for [Name] today at [Time]"**.
   - No duplicate record is added to the database.

### Step D: Test Unknown Person Detection
1. Have an unregistered colleague face the camera, or place an unknown image in front of the lens.
2. The HUD displays an **amber/red bounding box** with the label `Unknown (Score %)`.
3. The status banner confirms: **"Unknown Face Detected (Face not enrolled in database)"**.
4. No attendance is marked.

### Step E: Test Reports & CSV Export
1. Go to **Attendance Records** to filter by Date, Department, or Roll Number.
2. Click **"Export Filtered CSV"** to download the spreadsheet report.
3. Click **"Daily Sheet (Present/Absent)"** to download the complete classroom register showing all enrolled students with their Present or Absent status.
4. Go to **Reports & Stats** to inspect cumulative attendance percentage and exam eligibility (≥ 75%).

---

## 8. Running Automated Unit Tests

The project includes an automated test suite verifying all 10 core functional and security scenarios:

```powershell
python -m unittest discover -s tests
```

### Verified Test Cases:
1. `test_01_database_connection_and_seed`: Verifies DB schema, foreign keys, and 10 seeded students.
2. `test_02_successful_student_registration`: Verifies form submission and embedding pipeline.
3. `test_03_duplicate_student_registration`: Confirms duplicate Student IDs or Roll Numbers are blocked.
4. `test_04_duplicate_attendance_prevention`: Proves the database constraint blocks marking twice in 1 day.
5. `test_05_invalid_or_missing_frame_handling`: Verifies graceful error handling for missing/corrupted frames.
6. `test_06_registered_face_embedding_match`: Tests mathematical cosine similarity on enrolled vectors.
7. `test_07_unknown_face_rejection`: Confirms non-enrolled random embeddings are rejected.
8. `test_08_attendance_report_export`: Verifies Pandas CSV generation and download stream.
9. `test_09_daily_register_export`: Verifies the daily Present vs Absent classroom register CSV.
10. `test_10_student_deletion_and_erasure`: Verifies right-to-erasure GDPR/privacy compliance.

---

## 9. Troubleshooting & Common Issues

| Issue / Error | Root Cause | Solution |
|---|---|---|
| **"Webcam access failed" or black box in browser** | Browser permissions blocked or another application is using the camera. | 1. Click the camera icon in your browser address bar and choose "Allow".<br>2. Close applications using the camera (Zoom, Teams, Skype, Windows Camera app).<br>3. Refresh the page. |
| **"Face not detected during enrollment"** | Poor lighting, face too far from camera, or head tilted. | Ensure good room lighting, face the camera directly, and ensure your head fills ~50% of the video frame. |
| **"Multiple faces detected" error** | Another person is visible in the background during registration. | Ensure only the student is standing in front of the camera during enrollment. |
| **Cosine threshold too strict / not matching** | Default threshold is 0.38; differences in lighting or glasses might reduce cosine score. | Go to **Settings & Privacy** or use the threshold slider on the scanner page and adjust to `0.34` - `0.36`. |
| **Port 5000 already in use** | Another local server or process is running on port 5000. | Change `port=5000` to `port=5050` in `app.py` or run `python app.py`. |

---

## 10. Privacy, Ethical Safeguards & Vision Limitations

As emphasized in **Section 11 & 13 of the Workshop Report**:
- **Voluntary Informed Consent:** Biometric templates are generated strictly with student consent.
- **Isolated Storage:** Embeddings and face photos reside in protected local storage (`data/secure_storage/`), not exposed to public URLs.
- **Zero Cloud Transmission:** All deep learning inference runs 100% on-device on the host laptop.
- **Right to Erasure:** Administrators can permanently purge student biometric data via the **Reports** page.
- **Unrecognized ≠ Proof of Absence:** System failure or bad lighting is never treated as definitive absence. Authorized administrators can use the **Manual Attendance Override** modal to record attendance manually.
- **2D Webcam Limitations:** Standard laptop webcams capture 2D RGB imagery and do not possess 3D structured light or infrared depth sensors. Without specialized hardware, a printed 2D photograph could spoof a basic webcam system. An experimental sharpness/clarity heuristic is included in Settings to detect blurry/printed artifacts.

---

## 11. Directory Structure

```
face-attendance-system/
├── app.py                         # Flask web application entry point & API routes
├── config.py                      # System configuration, paths, and threshold constants
├── requirements.txt               # Compatible dependencies
├── README.md                      # Complete workshop documentation and guide
├── database/
│   ├── __init__.py
│   ├── db.py                      # SQLite database connection & query helpers
│   ├── schema.sql                 # Database table schema with unique constraints
│   └── seed.py                    # Populates 10 fictional students from workshop report
├── services/
│   ├── __init__.py
│   ├── face_recognition_service.py# OpenCV YuNet detection & SFace 128D embedding engine
│   └── attendance_service.py      # Business logic: enrollment, matching, reports, duplicates
├── models/
│   ├── yunet.onnx                 # OpenCV YuNet CNN Face Detector (ONNX)
│   ├── sface.onnx                 # OpenCV SFace Deep Face Recognizer (ONNX)
│   └── haarcascade_frontalface_default.xml # Auxiliary Haar Cascade
├── data/
│   ├── face_attendance.db         # SQLite database file
│   └── secure_storage/            # Protected non-public biometric storage
│       ├── embeddings/            # 128D .npy embedding arrays
│       └── photos/                # Student profile photos
├── templates/
│   ├── base.html                  # Responsive navy layout with sidebar & live clock
│   ├── login.html                 # Admin login portal
│   ├── dashboard.html             # Summary KPI cards & Chart.js visualizations
│   ├── register_student.html      # Enrollment form with webcam snapshot & upload tabs
│   ├── face_attendance.html       # Live webcam scanner HUD & real-time attendee feed
│   ├── attendance_records.html    # Attendance table, search, filters & manual override modal
│   ├── reports.html               # Percentage calculations, exam eligibility & erasure
│   └── settings.html              # Threshold slider, liveness heuristic & privacy policies
├── static/
│   ├── css/
│   │   └── style.css              # Custom styling (dark navy academic theme)
│   └── js/
│       └── attendance.js          # WebRTC video capture, HUD drawing & audio feedback
└── tests/
    ├── __init__.py
    └── test_attendance.py         # 10 automated unit test cases
```
