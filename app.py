import streamlit as st
import cv2
from mtcnn import MTCNN
import joblib
import pandas as pd
import io
from src.db import setup_database
from src.model import register_student, mark_attendance
from src.recognition import recognize_face_fast

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
# Initialize session state
# ----------------------------
if "page" not in st.session_state:
    st.session_state.page = "Student Registration"
if "run_recognition" not in st.session_state:
    st.session_state.run_recognition = False

# ----------------------------
# Sidebar navigation
# ----------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Student Registration", "Real-Time Recognition"],
    index=0 if st.session_state.page == "Student Registration" else 1
)

# ----------------------------
# PAGE 1: STUDENT REGISTRATION
# ----------------------------
if page == "Student Registration":
    st.session_state.page = "Student Registration"
    st.title("📚 Student Registration")

    matric = st.text_input("Matric Number")
    name = st.text_input("Full Name")

    if st.button("Register Student"):
        if matric and name:
            register_student(matric, name, cursor, conn)
            st.success(f"✅ {name} ({matric}) registered successfully!")
        else:
            st.error("Please enter both Matric Number and Name.")

# ----------------------------
# PAGE 2: REAL-TIME RECOGNITION
# ----------------------------
elif page == "Real-Time Recognition":
    st.session_state.page = "Real-Time Recognition"
    st.title("🕵️ Real-Time Attendance Recognition")

    # Buttons to start/stop recognition
    if st.button("Start Recognition"):
        st.session_state.run_recognition = True
    if st.button("Stop Recognition"):
        st.session_state.run_recognition = False

    FRAME_WINDOW = st.empty()

    if st.session_state.run_recognition:
        cap = cv2.VideoCapture(0)
        detector = MTCNN()

        while st.session_state.run_recognition and cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                continue

            # Convert BGR to RGB for MTCNN & display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                faces = detector.detect_faces(frame_rgb)
            except Exception as e:
                st.warning(f"Skipped frame due to detection error: {e}")
                continue

            for f in faces:
                x, y, w, h = f['box']
                x, y = max(0, x), max(0, y)
                face_crop = frame_rgb[y:y+h, x:x+w]

                # Recognize student (returns matric)
                label = recognize_face_fast(
                    face_crop, clf, le,
                    base_embeddings_dir="embeddings",
                    log_attendance=False
                )

                if label not in ["Unknown", "NoFace", "Error"]:
                    marked = mark_attendance(label, cursor, conn)
                    status = "Present" if marked else "AlreadyMarked"
                else:
                    status = "Unknown"

                # Pick color based on status
                color = (0, 255, 0) if status == "Present" else (0, 0, 255)

                # Draw bounding box + label on frame
                cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame_rgb, f"{label} ({status})",
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

            # Update the image in the Streamlit app
            FRAME_WINDOW.image(frame_rgb)

        cap.release()
        st.success("🛑 Recognition stopped.")

    # ----------------------------
    # Attendance Records (CSV Download)
    # ----------------------------
    st.subheader("📊 Attendance Records")
    cursor.execute("""
        SELECT a.matric, s.name, a.date, a.timestamp
        FROM attendance a
        JOIN students s ON a.matric = s.matric
        ORDER BY a.date DESC, a.timestamp DESC
    """)
    rows = cursor.fetchall()

    if rows:
        df = pd.DataFrame(rows, columns=["Matric", "Name", "Date", "Timestamp"])
        st.dataframe(df)

        # CSV Download
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Attendance (CSV)",
            data=csv_buffer.getvalue(),
            file_name="attendance_records.csv",
            mime="text/csv"
        )
    else:
        st.info("No attendance records found yet.")
