"""SQLite store for users, onboarding preferences, and interactions (warm-up)."""
import sqlite3
import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data/app.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            created_at TEXT,
            review_count INTEGER DEFAULT 0,
            is_cold_start INTEGER DEFAULT 1,
            preferences TEXT DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            gmap_id TEXT,
            rating REAL,
            timestamp TEXT,
            source TEXT DEFAULT 'explicit'
        );
    """)
    conn.commit()
    conn.close()


def upsert_user(user_id, name=None, preferences=None):
    conn = get_conn()
    existing = conn.execute("SELECT 1 FROM users WHERE user_id=?", (user_id,)).fetchone()
    if not existing:
        conn.execute(
            "INSERT INTO users (user_id, name, created_at, preferences) VALUES (?,?,?,?)",
            (user_id, name or user_id, datetime.now().isoformat(),
             json.dumps(preferences or {})),
        )
    elif preferences is not None or name is not None:
        if preferences is not None:
            conn.execute("UPDATE users SET preferences=? WHERE user_id=?",
                         (json.dumps(preferences), user_id))
        if name is not None:
            conn.execute("UPDATE users SET name=? WHERE user_id=?", (name, user_id))
    conn.commit()
    conn.close()


def log_interaction(user_id, gmap_id, rating, source="explicit"):
    conn = get_conn()
    conn.execute(
        "INSERT INTO interactions (user_id, gmap_id, rating, timestamp, source) VALUES (?,?,?,?,?)",
        (user_id, gmap_id, rating, datetime.now().isoformat(), source),
    )
    count = conn.execute("SELECT COUNT(*) FROM interactions WHERE user_id=?",
                         (user_id,)).fetchone()[0]
    conn.execute("UPDATE users SET review_count=?, is_cold_start=? WHERE user_id=?",
                 (count, 1 if count < 3 else 0, user_id))
    conn.commit()
    conn.close()


def get_user(user_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    try:
        d["preferences"] = json.loads(d.get("preferences") or "{}")
    except Exception:
        d["preferences"] = {}
    return d


def get_user_interactions(user_id, limit=100):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM interactions WHERE user_id=? ORDER BY timestamp DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


init_db()
