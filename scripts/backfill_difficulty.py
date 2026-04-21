"""
One-off migration: ensure remote Turso has a `difficulty` column on
completions, then backfill any NULL rows with a weighted random label.

Why this script connects directly to the remote URL instead of going
through scripts/db.py: the app uses libsql embedded-replica mode, which
replicates DML fine but does NOT push DDL (ALTER TABLE) back to the
primary. So the PRAGMA-gated migration in db.py runs on every container
start against a fresh replica — the column exists locally during the
session but never actually lands on Turso. For a schema fix we have to
hit the remote directly.

Weights for the backfill: easy=0.6, medium=0.3, hard=0.1 (user requested
easy highest, hard lowest; stats UI just shows the breakdown so the exact
numbers aren't load-bearing).

Usage:
    python scripts/backfill_difficulty.py              # dry run
    python scripts/backfill_difficulty.py --apply      # writes to remote
"""

import argparse
import os
import random
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import libsql_experimental as libsql


WEIGHTS = [("easy", 0.6), ("medium", 0.3), ("hard", 0.1)]


def pick_difficulty(rng: random.Random) -> str:
    r = rng.random()
    cum = 0.0
    for label, w in WEIGHTS:
        cum += w
        if r < cum:
            return label
    return WEIGHTS[-1][0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write updates (default: dry run)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    args = parser.parse_args()

    url = os.environ.get("TURSO_DATABASE_URL")
    token = os.environ.get("TURSO_AUTH_TOKEN")
    if not url or not token:
        print("TURSO_DATABASE_URL and TURSO_AUTH_TOKEN must be set", file=sys.stderr)
        return 1

    # Direct-to-remote connection (see module docstring for why).
    conn = libsql.connect(database=url, auth_token=token)

    cols = {r[1] for r in conn.execute("PRAGMA table_info(completions)").fetchall()}
    if "difficulty" not in cols:
        print("remote is missing difficulty column; adding it")
        if args.apply:
            conn.execute("ALTER TABLE completions ADD COLUMN difficulty TEXT")
            conn.commit()
        else:
            print("(dry run — column would be added; backfill preview uses all rows)")

    # After the ALTER we can safely select difficulty; on a dry run
    # without the column, fall back to selecting every row.
    if "difficulty" in cols or args.apply:
        rows = conn.execute(
            "SELECT id FROM completions WHERE difficulty IS NULL"
        ).fetchall()
    else:
        rows = conn.execute("SELECT id FROM completions").fetchall()
    total_null = len(rows)
    total_all = conn.execute("SELECT COUNT(*) FROM completions").fetchone()[0]
    print(f"completions with NULL difficulty: {total_null} / {total_all}")

    if total_null == 0:
        print("nothing to backfill")
        return 0

    rng = random.Random(args.seed)
    assignments = [(row[0], pick_difficulty(rng)) for row in rows]

    counts = {"easy": 0, "medium": 0, "hard": 0}
    for _, d in assignments:
        counts[d] += 1
    print(f"planned assignment: {counts}")

    if not args.apply:
        print("dry run — pass --apply to write")
        return 0

    for rid, diff in assignments:
        conn.execute("UPDATE completions SET difficulty = ? WHERE id = ?", (diff, rid))
    conn.commit()

    remaining = conn.execute(
        "SELECT COUNT(*) FROM completions WHERE difficulty IS NULL"
    ).fetchone()[0]
    print(f"applied {len(assignments)} updates; NULL remaining: {remaining}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
