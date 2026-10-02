import sqlite3
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "analysis_history.db"


def init_db():
    """Create the SQLite table used to store analysis history."""
    connection = sqlite3.connect(DB_PATH)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_type TEXT NOT NULL,
            input_value TEXT NOT NULL,
            title TEXT,
            domain TEXT,
            prediction TEXT,
            model_confidence REAL,
            source_status TEXT,
            credibility_score INTEGER,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.commit()
    connection.close()


def save_analysis(data):
    """Store one analysis result in SQLite."""
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    cursor.execute(
        """
        INSERT INTO analyses (
            input_type,
            input_value,
            title,
            domain,
            prediction,
            model_confidence,
            source_status,
            credibility_score,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data.get("input_type", "text"),
            data.get("input_value", ""),
            data.get("title", ""),
            data.get("domain", ""),
            data.get("prediction", ""),
            float(data.get("model_confidence", 0.0)),
            data.get("source_status", ""),
            int(data.get("credibility_score", 0)),
            datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    connection.commit()
    connection.close()


def get_history(limit=20):
    """Get recent saved analyses from the database."""
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT *
        FROM analyses
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    connection.close()
    return [dict(row) for row in rows]


if __name__ == "__main__":
    init_db()
    print(f"Database ready: {DB_PATH}")
