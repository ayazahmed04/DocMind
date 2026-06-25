# feedback_db.py
import sqlite3
from datetime import datetime

DB_PATH = "feedback.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  question TEXT,
                  answer TEXT,
                  rating INTEGER,  -- 1 for thumbs up, 0 for thumbs down
                  timestamp TEXT)''')
    conn.commit()
    conn.close()

def save_feedback(question, answer, rating):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO feedback (question, answer, rating, timestamp) VALUES (?, ?, ?, ?)",
              (question, answer, rating, datetime.now().isoformat()))
    conn.commit()
    conn.close()