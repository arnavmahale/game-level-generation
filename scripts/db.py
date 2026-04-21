"""
Turso (libSQL) access + schema for auth and leaderboard.

Uses libsql_experimental's embedded-replica mode: queries hit a local
SQLite file that's synced to the remote Turso database. Cheap reads,
strong consistency within the writing process. The local file is
rebuilt from the remote on every container start (HF's filesystem is
ephemeral), so we call conn.sync() once during init.
"""

import os
import threading
import logging
import bcrypt
import libsql_experimental as libsql

log = logging.getLogger(__name__)


_REPLICA_PATH = os.environ.get("TURSO_REPLICA_PATH", "genterrain.db")
# Per-thread connections: libsql/sqlite Connection objects are bound to the
# thread that created them, and Flask's dev server + gunicorn sync workers
# both dispatch requests across multiple threads.
_tls = threading.local()
_schema_lock = threading.Lock()
_schema_initialized = False


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT UNIQUE NOT NULL,
    password_hash   TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS completions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    model       TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
);
CREATE INDEX IF NOT EXISTS idx_completions_user_model ON completions(user_id, model);

CREATE TABLE IF NOT EXISTS endless_scores (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    score       INTEGER NOT NULL,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
);
CREATE INDEX IF NOT EXISTS idx_endless_user_score ON endless_scores(user_id, score DESC);
"""


def _connect():
    url = os.environ.get("TURSO_DATABASE_URL")
    token = os.environ.get("TURSO_AUTH_TOKEN")
    if not url or not token:
        raise RuntimeError("TURSO_DATABASE_URL and TURSO_AUTH_TOKEN must be set")
    conn = libsql.connect(database=_REPLICA_PATH, sync_url=url, auth_token=token)
    conn.sync()
    return conn


def _ensure_schema(conn):
    global _schema_initialized
    if _schema_initialized:
        return
    with _schema_lock:
        if _schema_initialized:
            return
        for stmt in filter(None, (s.strip() for s in SCHEMA.split(";"))):
            conn.execute(stmt)
        conn.commit()
        _schema_initialized = True


def get_conn():
    conn = getattr(_tls, "conn", None)
    if conn is None:
        conn = _connect()
        _tls.conn = conn
        _ensure_schema(conn)
    return conn


def _exec(sql, params=()):
    # Pull any remote changes so this thread's replica sees writes made by
    # other request threads (e.g. a leaderboard read after someone else
    # scored). Turso sync is fast (~100ms) and the app is low-QPS.
    conn = get_conn()
    try:
        conn.sync()
    except Exception as e:
        log.warning("sync() pre-read failed: %s", e)
    return conn.execute(sql, params)


def _exec_commit(sql, params=()):
    conn = get_conn()
    cur = conn.execute(sql, params)
    conn.commit()
    try:
        conn.sync()
    except Exception as e:
        log.warning("sync() post-write failed: %s", e)
    return cur


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


def create_user(username: str, password: str) -> int | None:
    try:
        cur = _exec_commit(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, hash_password(password)),
        )
        return cur.lastrowid
    except Exception:
        return None  # unique violation, etc.


def get_user_by_username(username: str):
    row = _exec(
        "SELECT id, username, password_hash FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    if row is None:
        return None
    return {"id": row[0], "username": row[1], "password_hash": row[2]}


def get_user_by_id(user_id: int):
    row = _exec(
        "SELECT id, username FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    if row is None:
        return None
    return {"id": row[0], "username": row[1]}


def record_completion(user_id: int, model: str):
    log.info("record_completion user_id=%s model=%s", user_id, model)
    cur = _exec_commit(
        "INSERT INTO completions (user_id, model) VALUES (?, ?)",
        (user_id, model),
    )
    log.info("record_completion inserted rowid=%s", getattr(cur, "lastrowid", None))


def record_endless_score(user_id: int, score: int):
    log.info("record_endless_score user_id=%s score=%s", user_id, score)
    cur = _exec_commit(
        "INSERT INTO endless_scores (user_id, score) VALUES (?, ?)",
        (user_id, int(score)),
    )
    log.info("record_endless_score inserted rowid=%s", getattr(cur, "lastrowid", None))


def stats_for_user(user_id: int):
    rows = _exec(
        "SELECT model, COUNT(*) FROM completions WHERE user_id = ? GROUP BY model",
        (user_id,),
    ).fetchall()
    completions = {"naive": 0, "bigram": 0, "vae": 0}
    for model, n in rows:
        if model in completions:
            completions[model] = n
    best = _exec(
        "SELECT COALESCE(MAX(score), 0) FROM endless_scores WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    return {
        "completions": completions,
        "endless_best": best[0] if best else 0,
    }


def leaderboard_vae(limit: int = 10):
    rows = _exec(
        """
        SELECT u.username, COUNT(c.id) AS n
        FROM users u
        JOIN completions c ON c.user_id = u.id
        WHERE c.model = 'vae'
        GROUP BY u.id
        ORDER BY n DESC, u.username ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [{"username": r[0], "count": r[1]} for r in rows]


def leaderboard_endless(limit: int = 10):
    rows = _exec(
        """
        SELECT u.username, MAX(s.score) AS best
        FROM users u
        JOIN endless_scores s ON s.user_id = u.id
        GROUP BY u.id
        ORDER BY best DESC, u.username ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [{"username": r[0], "score": r[1]} for r in rows]
