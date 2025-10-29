import sqlite3
import os

DB_NAME = "attendance.db"

def get_connection():
    """Always connect to the database file."""
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn, conn.cursor()

def setup_database():
    """Create database & tables if they don't exist."""
    conn, cursor = get_connection()

    # --- Create tables if not exist ---
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        matric TEXT PRIMARY KEY,
        name TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        matric TEXT,
        date TEXT,
        timestamp TEXT,
        PRIMARY KEY (matric, date),
        FOREIGN KEY (matric) REFERENCES students(matric)
    )
    """)

    conn.commit()

    # --- Optional: insert sample data for testing ---
    cursor.execute("SELECT COUNT(*) FROM students")
    if cursor.fetchone()[0] == 0:
        sample_students = [
            ("3986", "Akinnusi Mary Hellen"),
            ("1234", "John Doe"),
            ("0529", "akande soji")
        ]
        cursor.executemany("INSERT INTO students (matric, name) VALUES (?, ?)", sample_students)
        conn.commit()
        print("[+] Inserted sample students")

    print("[✓] Database setup complete and ready.")
    return conn, cursor
