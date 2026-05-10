import sqlite3
from datetime import datetime

DB_NAME = "watermark.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS processed_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        original_path TEXT NOT NULL,
        result_path TEXT,
        status TEXT DEFAULT 'processing',
        confidence REAL DEFAULT 0,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()


def save_image(original_path):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO processed_images
    (original_path, created_at)
    VALUES (?, ?)
    """, (original_path, datetime.now().isoformat()))

    image_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return image_id


def update_result(image_id, result_path, confidence):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE processed_images
    SET result_path = ?, status = ?, confidence = ?
    WHERE id = ?
    """, (result_path, "completed", confidence, image_id))

    conn.commit()
    conn.close()


def get_image(image_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM processed_images WHERE id=?", (image_id,))
    row = cursor.fetchone()

    conn.close()
    return row