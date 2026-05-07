# Sistem Absensi Pengenalan Wajah 🎓

Aplikasi absensi berbasis web menggunakan Flask dan pengenalan wajah OpenCV LBPH dengan deteksi YOLOv8 (opsional).

## Fitur Lengkap

- ✅ Pengenalan wajah real-time (LBPH + Liveness Check)
- ✅ Deteksi wajah dengan YOLOv8 atau Haar Cascade (fallback)
- ✅ Registrasi wajah 1-shot dengan webcam
- ✅ Mode absensi mandiri (viewer tanpa login)
- ✅ Absensi manual via NIM
- ✅ Notifikasi Telegram (foto + teks)
- ✅ Manajemen pengguna (mahasiswa/dosen/staff)
- ✅ Multi-role admin (admin / superadmin)
- ✅ Laporan & export Excel per bulan
- ✅ Filter absensi per tanggal
- ✅ Deteksi terlambat (batas waktu configurable)
- ✅ Training ulang model AI
- ✅ Desain akademik (hijau/biru)

## Instalasi

```bash
# 1. Clone/ekstrak project
cd attendance_system

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Opsional) Install YOLOv8 untuk deteksi wajah lebih akurat
pip install ultralytics
# Download model: yolov8n-face.pt (otomatis saat pertama kali dijalankan)

# 4. Jalankan aplikasi
python app.py
```

## Akses

- **Admin Panel:** http://localhost:5000/login
  - Username: `admin`
  - Password: `admin123`
- **Mode Absensi Mandiri:** http://localhost:5000/viewer

## Panduan Penggunaan

### 1. Daftarkan Wajah
Menu → Daftarkan Wajah → Isi nama & NIM → Posisikan wajah → Ambil Foto → Simpan

### 2. Training Model
Setelah mendaftarkan pengguna, klik "Training Ulang Model" di Dashboard atau Settings

### 3. Absensi
Menu → Absensi Wajah → Klik "Kenali Wajah" atau aktifkan Mode Otomatis

### 4. Setup Telegram (Opsional)
Settings → Masukkan Bot Token & Chat ID → Simpan

## Struktur Folder

```
attendance_system/
├── app.py              # Aplikasi utama Flask
├── requirements.txt    # Dependencies
├── attendance.db       # Database SQLite (auto-created)
├── models/             # Model AI tersimpan
├── static/
│   └── faces/          # Foto wajah & snapshot
└── templates/          # Template HTML
    ├── base.html
    ├── login.html
    ├── index.html
    ├── register.html
    ├── recognize.html
    ├── users.html
    ├── attendance.html
    ├── reports.html
    ├── admin_accounts.html
    ├── settings.html
    └── viewer.html
```

## Catatan Teknis

- **Database:** SQLite (tidak perlu install MySQL)
- **Face Recognition:** OpenCV LBPH (ringan, tidak butuh GPU)
- **Face Detection:** Haar Cascade (default) atau YOLOv8 (jika terinstall)
- **Liveness Detection:** Eye detection sederhana untuk anti-spoofing foto
- **Port default:** 5000

## Troubleshooting

**Kamera tidak terdeteksi:** Pastikan browser mengizinkan akses kamera (HTTPS atau localhost)

**Wajah tidak dikenali:** Coba training ulang model, pastikan pencahayaan cukup

**YOLOv8 error:** Hapus `# ` di requirements.txt dan jalankan `pip install ultralytics`
