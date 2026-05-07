from flask import Flask, render_template, request, jsonify, Response, session, redirect, url_for, flash, send_file
import cv2
import os
import numpy as np
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import base64
import json
import hashlib
import secrets
import time
import threading
import io
import zipfile
from functools import wraps

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# ── Config ──────────────────────────────────────────────────────────────────
TMP_DIR = "/tmp"

DB_PATH = os.path.join(TMP_DIR, "attendance.db")
MODELS_DIR = os.path.join(TMP_DIR, "models")
UPLOAD_FOLDER = os.path.join(TMP_DIR, "faces")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
YOLO_AVAILABLE = False

try:
    import requests as req_lib
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False

# ── Database ─────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            nim TEXT UNIQUE NOT NULL,
            role TEXT DEFAULT 'mahasiswa',
            face_data BLOB,
            face_image TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            nim TEXT,
            name TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            date TEXT,
            time TEXT,
            status TEXT DEFAULT 'hadir',
            method TEXT DEFAULT 'face',
            location TEXT,
            confidence REAL,
            photo TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'admin',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT,
            start_time TEXT,
            end_time TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    # Default admin
    pwd = hashlib.sha256('admin123'.encode()).hexdigest()
    c.execute("INSERT OR IGNORE INTO admins (username, password, role) VALUES (?, ?, ?)",
              ('admin', pwd, 'superadmin'))
    # Default settings
    defaults = [('telegram_token',''),('telegram_chat_id',''),
                ('late_threshold','08:00'),('app_name','Sistem Absensi Wajah'),
                ('institution','Universitas')]
    for k, v in defaults:
        c.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (k, v))
    conn.commit()
    conn.close()

def get_setting(key, default=''):
    conn = get_db()
    row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    conn.close()
    return row['value'] if row else default

# ── Face Recognition ─────────────────────────────────────────────────────────
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
MODEL_FILE = os.path.join(MODELS_DIR, 'face_model.yml')
LABELS_FILE = os.path.join(MODELS_DIR, 'labels.json')
model_loaded = False
label_map = {}  # id -> user_id

def train_model():
    global face_recognizer, model_loaded, label_map
    conn = get_db()
    users = conn.execute("SELECT id, face_data FROM users WHERE face_data IS NOT NULL").fetchall()
    conn.close()
    if not users:
        return False
    faces, labels = [], []
    label_map = {}
    for idx, u in enumerate(users):
        arr = np.frombuffer(u['face_data'], dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            faces.append(cv2.resize(img, (200, 200)))
            labels.append(idx)
            label_map[idx] = u['id']
    if not faces:
        return False
    face_recognizer.train(faces, np.array(labels))
    face_recognizer.save(MODEL_FILE)
    with open(LABELS_FILE, 'w') as f:
        json.dump({str(k): v for k, v in label_map.items()}, f)
    model_loaded = True
    return True

def load_model():
    global face_recognizer, model_loaded, label_map
    if os.path.exists(MODEL_FILE) and os.path.exists(LABELS_FILE):
        face_recognizer.read(MODEL_FILE)
        with open(LABELS_FILE) as f:
            label_map = {int(k): v for k, v in json.load(f).items()}
        model_loaded = True

def detect_faces_yolo(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80,80))

def check_liveness(frame, x, y, w, h):
    """Simple blink/eye-open liveness check"""
    roi = frame[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_roi, 1.1, 3, minSize=(20,20))
    return len(eyes) >= 1  # at least one eye visible

def recognize_face(frame):
    if not model_loaded:
        return None, 0, False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces_yolo(frame)
    if len(faces) == 0:
        return None, 0, False
    x, y, w, h = faces[0][:4]
    face_roi = gray[y:y+h, x:x+w]
    face_roi = cv2.resize(face_roi, (200, 200))
    try:
        label, confidence = face_recognizer.predict(face_roi)
        live = check_liveness(frame, x, y, w, h)
        if confidence < 80:
            user_id = label_map.get(label)
            return user_id, confidence, live
    except:
        pass
    return None, 0, False

# ── Auth ──────────────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'admin_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def superadmin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'admin_id' not in session:
            return redirect(url_for('login'))
        if session.get('admin_role') != 'superadmin':
            flash('Akses ditolak', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated

# ── Telegram ──────────────────────────────────────────────────────────────────
def send_telegram(message, photo_path=None):
    token = get_setting('telegram_token')
    chat_id = get_setting('telegram_chat_id')
    if not token or not chat_id or not REQUESTS_AVAILABLE:
        return False
    try:
        if photo_path and os.path.exists(photo_path):
            url = f"https://api.telegram.org/bot{token}/sendPhoto"
            with open(photo_path, 'rb') as ph:
                req_lib.post(url, data={'chat_id': chat_id, 'caption': message},
                             files={'photo': ph}, timeout=5)
        else:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            req_lib.post(url, json={'chat_id': chat_id, 'text': message,
                                    'parse_mode': 'HTML'}, timeout=5)
        return True
    except:
        return False

# ── Routes: Auth ─────────────────────────────────────────────────────────────
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username','')
        password = hashlib.sha256(request.form.get('password','').encode()).hexdigest()
        conn = get_db()
        admin = conn.execute("SELECT * FROM admins WHERE username=? AND password=?",
                             (username, password)).fetchone()
        conn.close()
        if admin:
            session['admin_id'] = admin['id']
            session['admin_username'] = admin['username']
            session['admin_role'] = admin['role']
            return redirect(url_for('index'))
        flash('Username atau password salah', 'error')
    return render_template('login.html', app_name=get_setting('app_name','Sistem Absensi'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ── Routes: Dashboard ─────────────────────────────────────────────────────────
@app.route('/')
@login_required
def index():
    conn = get_db()
    today = datetime.now().strftime('%Y-%m-%d')
    stats = {
        'total_users': conn.execute("SELECT COUNT(*) FROM users").fetchone()[0],
        'today_attendance': conn.execute("SELECT COUNT(*) FROM attendance WHERE date=?", (today,)).fetchone()[0],
        'total_attendance': conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0],
        'model_trained': model_loaded,
    }
    recent = conn.execute(
        "SELECT * FROM attendance ORDER BY timestamp DESC LIMIT 10").fetchall()
    conn.close()
    return render_template('index.html', stats=stats, recent=recent,
                           app_name=get_setting('app_name','Sistem Absensi'),
                           now=datetime.now())

# ── Routes: Users ─────────────────────────────────────────────────────────────
@app.route('/users')
@login_required
def users():
    conn = get_db()
    all_users = conn.execute("SELECT * FROM users ORDER BY name").fetchall()
    conn.close()
    return render_template('users.html', users=all_users,
                           app_name=get_setting('app_name'))

@app.route('/users/delete/<int:uid>', methods=['POST'])
@login_required
def delete_user(uid):
    conn = get_db()
    conn.execute("DELETE FROM users WHERE id=?", (uid,))
    conn.execute("DELETE FROM attendance WHERE user_id=?", (uid,))
    conn.commit()
    conn.close()
    train_model()
    flash('User berhasil dihapus', 'success')
    return redirect(url_for('users'))

# ── Routes: Register ──────────────────────────────────────────────────────────
@app.route('/register', methods=['GET','POST'])
@login_required
def register():
    if request.method == 'POST':
        name = request.form.get('name','').strip()
        nim  = request.form.get('nim','').strip()
        role = request.form.get('role','mahasiswa')
        image_data = request.form.get('image_data','')
        if not name or not nim or not image_data:
            return jsonify({'success': False, 'message': 'Data tidak lengkap'})
        try:
            header, encoded = image_data.split(',', 1)
            img_bytes = base64.b64decode(encoded)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80,80))
            if len(faces) == 0:
                return jsonify({'success': False, 'message': 'Wajah tidak terdeteksi'})
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200,200))
            # Encode face as PNG bytes
            _, buf = cv2.imencode('.png', face_roi)
            face_blob = buf.tobytes()
            # Save face image
            face_img_path = f"static/faces/{nim}.jpg"
            cv2.imwrite(face_img_path, frame[y:y+h, x:x+w])
            conn = get_db()
            conn.execute(
                "INSERT OR REPLACE INTO users (name, nim, role, face_data, face_image) VALUES (?,?,?,?,?)",
                (name, nim, role, face_blob, face_img_path))
            conn.commit()
            conn.close()
            train_model()
            return jsonify({'success': True, 'message': f'Registrasi {name} berhasil!'})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
    return render_template('register.html', app_name=get_setting('app_name'))

# ── Routes: Recognize / Attendance ───────────────────────────────────────────
@app.route('/recognize')
@login_required
def recognize():
    return render_template('recognize.html', app_name=get_setting('app_name'))

@app.route('/api/recognize', methods=['POST'])
@login_required
def api_recognize():
    data = request.json
    image_data = data.get('image','')
    if not image_data:
        return jsonify({'success': False, 'message': 'No image'})
    try:
        header, encoded = image_data.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        user_id, confidence, live = recognize_face(frame)
        if user_id is None:
            return jsonify({'success': False, 'message': 'Wajah tidak dikenali'})
        if not live:
            return jsonify({'success': False, 'message': 'Liveness check gagal - gunakan wajah asli'})
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        if not user:
            conn.close()
            return jsonify({'success': False, 'message': 'User tidak ditemukan'})
        # Check duplicate today
        today = datetime.now().strftime('%Y-%m-%d')
        exists = conn.execute(
            "SELECT id FROM attendance WHERE user_id=? AND date=?", (user_id, today)).fetchone()
        if exists:
            conn.close()
            return jsonify({'success': False,
                            'message': f'{user["name"]} sudah absen hari ini',
                            'duplicate': True})
        now = datetime.now()
        time_str = now.strftime('%H:%M:%S')
        # Save snapshot
        snap_path = f"static/faces/snap_{user['nim']}_{today}.jpg"
        cv2.imwrite(snap_path, frame)
        conn.execute(
            "INSERT INTO attendance (user_id, nim, name, date, time, status, method, confidence, photo) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (user_id, user['nim'], user['name'], today, time_str, 'hadir', 'face',
             float(confidence), snap_path))
        conn.commit()
        conn.close()
        # Telegram notification
        late_th = get_setting('late_threshold','08:00')
        status_msg = '⚠️ TERLAMBAT' if time_str[:5] > late_th else '✅ HADIR'
        send_telegram(
            f"<b>{status_msg}</b>\n👤 {user['name']}\n🆔 {user['nim']}\n🕐 {time_str}\n📅 {today}",
            snap_path)
        return jsonify({'success': True,
                        'message': f'Absensi {user["name"]} berhasil!',
                        'name': user['name'], 'nim': user['nim'],
                        'time': time_str, 'confidence': round(100 - confidence, 1)})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# Manual attendance
@app.route('/api/attendance/manual', methods=['POST'])
@login_required
def manual_attendance():
    nim = request.form.get('nim','').strip()
    status = request.form.get('status','hadir')
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE nim=?", (nim,)).fetchone()
    if not user:
        conn.close()
        return jsonify({'success': False, 'message': 'NIM tidak ditemukan'})
    today = datetime.now().strftime('%Y-%m-%d')
    time_str = datetime.now().strftime('%H:%M:%S')
    conn.execute(
        "INSERT INTO attendance (user_id, nim, name, date, time, status, method) VALUES (?,?,?,?,?,?,?)",
        (user['id'], nim, user['name'], today, time_str, status, 'manual'))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': f'Absensi manual {user["name"]} berhasil'})

# ── Routes: Attendance ────────────────────────────────────────────────────────
@app.route('/attendance')
@login_required
def attendance():
    conn = get_db()
    date_filter = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    records = conn.execute(
        "SELECT * FROM attendance WHERE date=? ORDER BY timestamp DESC", (date_filter,)).fetchall()
    conn.close()
    return render_template('attendance.html', records=records, date_filter=date_filter,
                           app_name=get_setting('app_name'))

@app.route('/attendance/delete/<int:aid>', methods=['POST'])
@login_required
def delete_attendance(aid):
    conn = get_db()
    conn.execute("DELETE FROM attendance WHERE id=?", (aid,))
    conn.commit()
    conn.close()
    flash('Data absensi dihapus', 'success')
    return redirect(url_for('attendance'))

# ── Routes: Reports ───────────────────────────────────────────────────────────
@app.route('/reports')
@login_required
def reports():
    conn = get_db()
    month = request.args.get('month', datetime.now().strftime('%Y-%m'))
    records = conn.execute(
        "SELECT * FROM attendance WHERE date LIKE ? ORDER BY date DESC, time DESC",
        (f"{month}%",)).fetchall()
    # Summary per user
    summary = conn.execute(
        "SELECT nim, name, COUNT(*) as total, SUM(CASE WHEN status='hadir' THEN 1 ELSE 0 END) as hadir "
        "FROM attendance WHERE date LIKE ? GROUP BY nim", (f"{month}%",)).fetchall()
    conn.close()
    return render_template('reports.html', records=records, summary=summary,
                           month=month, app_name=get_setting('app_name'))

@app.route('/reports/export')
@login_required
def export_report():
    month = request.args.get('month', datetime.now().strftime('%Y-%m'))
    conn = get_db()
    df = pd.read_sql_query(
        "SELECT nim, name, date, time, status, method FROM attendance WHERE date LIKE ? ORDER BY date, nim",
        conn, params=(f"{month}%",))
    conn.close()
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Absensi')
    output.seek(0)
    return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                     as_attachment=True, download_name=f'absensi_{month}.xlsx')

# ── Routes: Admin Accounts ────────────────────────────────────────────────────
@app.route('/admins')
@superadmin_required
def admin_accounts():
    conn = get_db()
    admins = conn.execute("SELECT id, username, role, created_at FROM admins ORDER BY id").fetchall()
    conn.close()
    return render_template('admin_accounts.html', admins=admins, app_name=get_setting('app_name'))

@app.route('/admins/add', methods=['POST'])
@superadmin_required
def add_admin():
    username = request.form.get('username','').strip()
    password = request.form.get('password','')
    role     = request.form.get('role','admin')
    if not username or not password:
        flash('Username dan password wajib diisi', 'error')
        return redirect(url_for('admin_accounts'))
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    conn = get_db()
    try:
        conn.execute("INSERT INTO admins (username, password, role) VALUES (?,?,?)",
                     (username, pwd_hash, role))
        conn.commit()
        flash(f'Admin {username} berhasil ditambah', 'success')
    except:
        flash('Username sudah ada', 'error')
    conn.close()
    return redirect(url_for('admin_accounts'))

@app.route('/admins/delete/<int:aid>', methods=['POST'])
@superadmin_required
def delete_admin(aid):
    if aid == session.get('admin_id'):
        flash('Tidak bisa hapus akun sendiri', 'error')
        return redirect(url_for('admin_accounts'))
    conn = get_db()
    conn.execute("DELETE FROM admins WHERE id=?", (aid,))
    conn.commit()
    conn.close()
    flash('Admin dihapus', 'success')
    return redirect(url_for('admin_accounts'))

# ── Routes: Settings ──────────────────────────────────────────────────────────
@app.route('/settings', methods=['GET','POST'])
@superadmin_required
def settings():
    if request.method == 'POST':
        keys = ['app_name','institution','telegram_token','telegram_chat_id','late_threshold']
        conn = get_db()
        for k in keys:
            v = request.form.get(k,'')
            conn.execute("INSERT OR REPLACE INTO settings (key,value) VALUES (?,?)", (k, v))
        conn.commit()
        conn.close()
        flash('Pengaturan disimpan', 'success')
    conn = get_db()
    rows = conn.execute("SELECT * FROM settings").fetchall()
    conn.close()
    cfg = {r['key']: r['value'] for r in rows}
    return render_template('settings.html', cfg=cfg, app_name=cfg.get('app_name','Sistem Absensi'))

@app.route('/api/train', methods=['POST'])
@login_required
def api_train():
    ok = train_model()
    return jsonify({'success': ok, 'message': 'Model ditraining ulang' if ok else 'Tidak ada data wajah'})

# ── Viewer: Self-service attendance ─────────────────────────────────────────
@app.route('/viewer')
def viewer():
    return render_template('viewer.html', app_name=get_setting('app_name'))

@app.route('/api/viewer/recognize', methods=['POST'])
def viewer_recognize():
    return api_recognize()  # same logic


# Init app
init_db()
load_model()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
