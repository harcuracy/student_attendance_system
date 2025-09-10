from src.db import setup_database
from src.model import register_student
from src.recognition import recognize_multiple_faces
import joblib

# Setup database
conn, cursor = setup_database()

# Load trained models
clf = joblib.load("models/svm_clf.pkl")
le = joblib.load("models/label_encoder.pkl")

# Register students
register_student("2001", "John Doe", cursor, conn)
register_student("0540", "Jame isaac", cursor, conn)
register_student("0529", "akande soji", cursor, conn)

# Run recognition
result = recognize_multiple_faces("test_image/24.jpeg", clf, le,cursor,conn,
                             top_k=3, verify_ratio=0.5, sim_threshold=0.4)
cleaned = [(str(label), status) for label, status in result]
print("Final:", cleaned)