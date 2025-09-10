import cv2
from mtcnn import MTCNN
from src.db import get_connection
from src.model import mark_attendance
from src.recognition import recognize_face_fast
import joblib
from src.db import setup_database



# Load models
clf = joblib.load("models/svm_clf.pkl")
le = joblib.load("models/label_encoder.pkl")

# Connect to database
conn, cursor = setup_database()

# Initialize face detector
detector = MTCNN()

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector.detect_faces(frame)
    for f in faces:
        x, y, w, h = f['box']
        x, y = max(0, x), max(0, y)
        face_crop = frame[y:y+h, x:x+w]

        # Recognize face
        label = recognize_face_fast(face_crop, clf, le, log_attendance=False)

        # Mark attendance
        if label not in ["Unknown", "NoFace", "Error"]:
            marked = mark_attendance(label, cursor, conn)
            status = "Present" if marked else "AlreadyMarked"
        else:
            status = "Unknown"

        # Draw bounding box and label
        color = (0, 255, 0) if status == "Present" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{label} ({status})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Real-Time Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
