import os
import sqlite3
from typing import Optional, Dict
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DB_PATH = os.path.join(BASE_DIR, "data", "auth.db")

def _get_conn():
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = _get_conn()
    try:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
    finally:
        conn.close()

def create_user(email: str, name: str, password: str) -> bool:
    conn = _get_conn()
    try:
        conn.execute("INSERT INTO users (email, name, password_hash) VALUES (?, ?, ?)",
                     (email.strip().lower(), name.strip(), generate_password_hash(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user_by_email(email: str) -> Optional[Dict]:
    conn = _get_conn()
    try:
        cur = conn.execute("SELECT id, email, name, password_hash FROM users WHERE email = ?", (email.strip().lower(),))
        row = cur.fetchone()
        if not row:
            return None
        return {"id": row[0], "email": row[1], "name": row[2], "password_hash": row[3]}
    finally:
        conn.close()

def get_user_by_id(user_id: int) -> Optional[Dict]:
    conn = _get_conn()
    try:
        cur = conn.execute("SELECT id, email, name FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {"id": row[0], "email": row[1], "name": row[2]}
    finally:
        conn.close()

def verify_password(user: Dict, password: str) -> bool:
    return check_password_hash(user["password_hash"], password)
