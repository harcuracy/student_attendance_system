from flask import Flask, render_template, Response, send_file, request
import cv2
from mtcnn import MTCNN
import joblib
import pandas as pd
import io
from datetime import datetime, date
import os

from src.db import setup_database
from src.model import mark_attendance
from src.recognition import recognize_face_fast

os.environ["TF_USE_LEGACY_KERAS"] = "0"
app = Flask(__name__)

# ----------------------------
# Load models
# ----------------------------
clf = joblib.load("models/svm_clf.pkl")
le = joblib.load("models/label_encoder.pkl")

# ----------------------------
# Connect to database
# ----------------------------
conn, cursor = setup_database()

# ----------------------------
# Camera globals
# ----------------------------
camera = None
detector = MTCNN()
camera_running = False

# ----------------------------
# Camera control functions
# ----------------------------
def start_camera():
    global camera, camera_running
    if camera is None:
        camera = cv2.VideoCapture(0)
    camera_running = True

def stop_camera():
    global camera, camera_running
    camera_running = False
    if camera and camera.isOpened():
        camera.release()
        camera = None

# ----------------------------
# Video feed generator
# ----------------------------
def generate_frames():
    global camera_running, camera
    while camera_running:
        success, frame = camera.read()
        if not success:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            faces = detector.detect_faces(frame_rgb)
        except Exception:
            continue

        for f in faces:
            x, y, w, h = f['box']
            x, y = max(0, x), max(0, y)
            face_crop = frame_rgb[y:y+h, x:x+w]

            label = recognize_face_fast(face_crop, clf, le,
                                        base_embeddings_dir="embeddings",
                                        log_attendance=False)

            if label not in ["Unknown", "NoFace", "Error"]:
                marked = mark_attendance(label, cursor, conn)
                status = "Present" if marked else "AlreadyMarked"
            else:
                status = "Unknown"

            color = (0, 255, 0) if status == "Present" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} ({status})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ----------------------------
# Routes
# ----------------------------
@app.route('/')
def dashboard():
    cursor.execute("""
        SELECT s.name, a.matric, a.date
        FROM attendance a
        JOIN students s ON a.matric = s.matric
        ORDER BY a.date DESC
    """)
    rows = cursor.fetchall()
    attendance = [{"name": r[0], "matric": r[1], "date": r[2]} for r in rows]
    return render_template('dashboard.html', attendance=attendance)

@app.route('/recognition')
def recognition():
    start_camera()
    return render_template('recognition.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera', methods=['POST', 'GET'])
def stop_camera_route():
    stop_camera()
    return "Camera stopped"

@app.route('/download_csv')
def download_csv():
    cursor.execute("""
        SELECT a.matric, s.name, a.date
        FROM attendance a
        JOIN students s ON a.matric = s.matric
        ORDER BY a.date DESC
    """)
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=["Matric", "Name", "Date"])
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return send_file(
        io.BytesIO(buffer.getvalue().encode()),
        as_attachment=True,
        download_name=f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mimetype='text/csv'
    )

# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
