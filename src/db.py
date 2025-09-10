import sqlite3
import os

DB_NAME = "attendance.db"

def get_connection():
    """Always connect to the database file."""
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    return conn, conn.cursor()

def setup_database():
    """Create database & tables only if file does not exist."""
    first_time = not os.path.exists(DB_NAME)
    conn, cursor = get_connection()

    if first_time:
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
        print("[+] Database created and tables ready")
    else:
        print("[i] Database already exists, skipping setup")

    return conn, cursor
