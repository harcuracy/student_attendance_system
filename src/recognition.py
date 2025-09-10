import os
import cv2
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from src.model import mark_attendance

detector = MTCNN()

def recognize_face_fast(img_input, clf, le,
                        top_k=3, verify_ratio=0.6, sim_threshold=0.6,
                        base_embeddings_dir="embeddings",
                        log_attendance=False, cursor=None, conn=None):
    try:
        emb_objs = DeepFace.represent(
            img_path=img_input, model_name="Facenet", enforce_detection=False
        )
        if not emb_objs:
            return "NoFace"

        test_emb = np.array(emb_objs[0]["embedding"]).reshape(1, -1)
    except Exception as e:
        print(f"[!] Error embedding: {e}")
        return "Error"

    # SVM prediction
    probs = clf.predict_proba(test_emb)[0]
    top_idx = np.argsort(probs)[::-1][:max(1, top_k)]

    # Convert indices back to labels (matric numbers)
    top_labels = le.inverse_transform(top_idx)

    best_label, best_ratio = None, -1.0
    for label in top_labels:
        emb_files = [f for f in os.listdir(base_embeddings_dir) if f.startswith(label)]
        if not emb_files:
            continue

        gallery_embs = [np.load(os.path.join(base_embeddings_dir, f)) for f in emb_files]
        gallery_embs = np.array(gallery_embs)

        sims = cosine_similarity(test_emb, gallery_embs)[0]
        ratio = np.mean(sims >= sim_threshold)

        if ratio > best_ratio:
            best_ratio, best_label = ratio, label

    if best_label and best_ratio >= verify_ratio:
        if log_attendance and cursor and conn:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("INSERT INTO attendance VALUES (?, ?, ?)", (best_label, ts.split()[0], ts))
            conn.commit()
        return best_label  # ✅ returns matric

    return "Unknown"




def recognize_multiple_faces(img_path, clf, le, cursor, conn,
                            top_k=3, verify_ratio=0.6, sim_threshold=0.6,
                            base_embeddings_dir="embeddings"):
    """
    Detects all faces in the image and recognizes them.
    Marks attendance for recognized students if not already marked.
    Returns a list of (label, status) tuples.
    """
    img = cv2.imread(img_path)
    if img is None:
        return [("Error", "Cannot read image")]

    faces = detector.detect_faces(img)
    if not faces:
        return [("NoFace", "None")]

    results = []
    for f in faces:
        x, y, w, h = f['box']
        x, y = max(0, x), max(0, y)  # Fix negative coordinates
        face_crop = img[y:y+h, x:x+w]

        # Recognize face
        label = recognize_face_fast(
            face_crop, clf, le,
            top_k=top_k,
            verify_ratio=verify_ratio,
            sim_threshold=sim_threshold,
            base_embeddings_dir=base_embeddings_dir,
            log_attendance=False  # we handle attendance below
        )

        # Determine attendance status
        if label not in ["Unknown", "NoFace", "Error"]:
            marked = mark_attendance(label, cursor, conn)
            status = "Present" if marked else "AlreadyMarked"
        else:
            status = "Unknown"

        results.append((label, status))

    return results
