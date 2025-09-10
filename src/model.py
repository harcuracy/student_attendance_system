from datetime import datetime, date

def register_student(matric, name, cursor, conn):
    cursor.execute("INSERT OR IGNORE INTO students (matric, name) VALUES (?, ?)", (matric, name))
    conn.commit()
    print(f"[+] Registered {matric} - {name}")

def mark_attendance(matric, cursor, conn):
    today = date.today().isoformat()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("SELECT name FROM students WHERE matric=?", (matric,))
    student = cursor.fetchone()
    if not student:
        print(f"[!] {matric} not found in students table")
        return False

    cursor.execute("SELECT * FROM attendance WHERE matric=? AND date=?", (matric, today))
    if cursor.fetchone() is None:
        cursor.execute("INSERT INTO attendance (matric, date, timestamp) VALUES (?,?,?)", (matric, today, ts))
        conn.commit()
        print(f"[+] {matric} ({student[0]}) marked present at {ts}")
        return True
    else:
        print(f"[i] {matric} already marked present today")
        return False
