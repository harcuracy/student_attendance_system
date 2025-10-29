# 🎓 Student Attendance System (AI Face Recognition)

This is an AI-powered **Student Attendance System** built with **Python, TensorFlow, DeepFace, MTCNN, and OpenCV**.  
It detects and recognizes student faces in real time and automatically records attendance.

---

## 🚀 Features

✅ Real-time face detection and recognition  
✅ Works with webcam or uploaded images  
✅ Detects unknown faces  
✅ Uses DeepFace + MTCNN for accuracy  
✅ Flask-based web interface (optional Streamlit version)  
✅ Easy to set up and run locally  

---

## 🛠️ Tech Stack

- Python 3.10+
- TensorFlow 2.15.0  
- DeepFace 0.0.95  
- MTCNN 1.0.0  
- OpenCV 4.11.0.86  
- Scikit-learn 1.7.2  
- Flask 3.1.2  
- Streamlit 1.50.0  

---

## ⚙️ Setup Instructions

Follow these steps to install and run the system locally.

### 1️⃣ Install uv (Fast Python Package Manager)

```bash
pip install uv
```

### 2️⃣ Clone the repository

```bash
git clone https://github.com/harcuracy/student_attendance_system.git
cd student_attendance_system
```

### 3️⃣ Create a virtual environment (Python 3.11)

```bash
uv venv --python 3.11
```

### 4️⃣ Activate the virtual environment

```bash
.venv\Scripts\activate      # On Windows
# or
source .venv/bin/activate     # On macOS/Linux
```

### 5️⃣ Install all dependencies

```bash
uv pip install -r requirements.txt
```

---

## ▶️ Running the App

### 🔹 Python Version

```bash
cd student_attendance_system
python app.py
```

Then open your browser and visit:  
👉 [http://localhost:5000](http://localhost:5000)


## 📦 Example `requirements.txt`

These versions are confirmed to work together (Python 3.10):

```
tensorflow==2.15.0
numpy==1.26.4
keras-facenet==0.3.2
deepface==0.0.93
mtcnn==1.0.0
scikit-learn==1.6.1
pandas==2.2.3
joblib==1.4.2

```

---

## 🧠 How It Works

1. **MTCNN** detects faces in the video frame.  
2. **DeepFace / FaceNet** extracts facial embeddings.  
3. **SVM Classifier** identifies each face.  
4. Attendance is logged automatically in a CSV or database.  

---

## 👩‍💻 Author

**Akande soji (Irewolede)**  
AI/ML Engineer — Bamidele Olumilua University of Education, Science, and Technology  
📧 [akandesoji4christgmail.com](mailto:akandesoji4christ@gmail.com)

---

## 🤝 Contributing

Pull requests are welcome!  
If you’d like to improve the UI, add a new model, or enhance accuracy, fork the repo and submit a PR.

---

## 🪪 License

This project is open source and available under the **MIT License**.
