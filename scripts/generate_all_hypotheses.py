"""
scripts/generate_all_hypotheses.py

Backfill research hypotheses for every disease in the knowledge graph.

Loops through all diseases in data/bioiq.db, skips any that already have a
cached hypothesis file in data/processed/hypotheses/, and calls
synthesis.engine.generate_hypotheses for the rest — one Claude API call each,
with a 2-second delay between calls to stay clear of rate limits.

Progress is logged per disease and a summary of which diseases succeeded,
failed, or were skipped is written to data/processed/hypothesis_summary.json
(checkpointed after each disease, so an interrupted run keeps its progress).

IMPORTANT: each generated disease is a real (paid) Claude API call (~$0.10).
Use --dry-run to preview what would be generated, and --limit N to cap how many
diseases are processed in one run.

Usage:
    python scripts/generate_all_hypotheses.py                 # generate all missing
    python scripts/generate_all_hypotheses.py --dry-run       # preview only, no API calls
    python scripts/generate_all_hypotheses.py --limit 5       # only the first 5 missing
    python scripts/generate_all_hypotheses.py --refresh       # regenerate even if cached
    python scripts/generate_all_hypotheses.py --delay 5       # longer pause between calls
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Engine paths are relative (data/processed/...), so anchor the working
# directory to the project root regardless of where we're invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the engine's own cache directory so the "already exists?" check matches
# exactly where generate_hypotheses writes.
from synthesis.engine import HYPOTHESES_DIR, generate_hypotheses

DB_PATH = Path("data/bioiq.db")
SUMMARY_PATH = Path("data/processed/hypothesis_summary.json")
DEFAULT_DELAY_SECONDS = 2

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("generate_all_hypotheses")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    """Current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def load_disease_names() -> list[str]:
    """
    Return every disease name in data/bioiq.db, alphabetically.

    Returns:
        List of canonical disease names (empty if the DB is missing/unreadable).
    """
    if not DB_PATH.exists():
        logger.error("Database not found at %s — run the seed/graph build first.", DB_PATH)
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("SELECT name FROM disease ORDER BY name").fetchall()
        conn.close()
    except sqlite3.Error as e:
        logger.error("Could not read diseases from %s: %s", DB_PATH, e)
        return []
    return [r[0] for r in rows]


def _hypothesis_path(disease_name: str) -> Path:
    """
    Path where generate_hypotheses caches this disease's file.

    Mirrors the engine's own key: spaces and slashes become underscores.
    """
    safe_name = disease_name.replace(" ", "_").replace("/", "_")
    return HYPOTHESES_DIR / f"{safe_name}.json"


def write_summary(summary: dict) -> None:
    """Persist the run summary to data/processed/hypothesis_summary.json."""
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(*, dry_run: bool, limit: int | None, delay: float, refresh: bool) -> dict:
    """
    Generate hypotheses for every disease that doesn't already have a file.

    Args:
        dry_run: If True, only report what would be generated — no API calls.
        limit:   Cap on how many missing diseases to process (None = all).
        delay:   Seconds to sleep between successive API calls.
        refresh: If True, regenerate every disease even if it already has a file.

    Returns:
        The summary dict (also written to disk unless dry_run).
    """
    names = load_disease_names()
    started = _now()
    t0 = time.time()

    # Partition into those already cached vs. those needing generation.
    already_have = [n for n in names if _hypothesis_path(n).exists()]
    todo = names if refresh else [n for n in names if not _hypothesis_path(n).exists()]
    if limit is not None:
        todo = todo[:limit]

    summary: dict = {
        "started_at": started,
        "finished_at": None,
        "duration_seconds": None,
        "total_diseases": len(names),
        "already_had": len(already_have),
        "attempted": len(todo),
        "refresh": refresh,
        "succeeded": [],
        "failed": [],       # list of {disease, error}
        "skipped": already_have if not refresh else [],
        "per_disease": {},
        "total_cost_usd": 0.0,
    }

    if dry_run:
        print(f"DRY RUN — {len(names)} diseases in DB: "
              f"{len(already_have)} already have hypotheses, "
              f"{len(todo)} would be generated"
              f"{' (--refresh: regenerating all)' if refresh else ''}.")
        for i, name in enumerate(todo, 1):
            print(f"  {i:2}. would generate: {name}")
        if not refresh and already_have:
            print(f"\n  ({len(already_have)} skipped as already cached)")
        return summary

    if not todo:
        logger.info("Nothing to do — all %d diseases already have hypotheses.", len(names))

    for i, name in enumerate(todo, start=1):
        logger.info("[%d/%d] Generating hypotheses for %r", i, len(todo), name)
        try:
            result = generate_hypotheses(name, refresh=refresh)
        except Exception as e:  # noqa: BLE001 — never let one disease abort the batch
            msg = f"{type(e).__name__}: {e}"
            logger.error("  %s raised: %s", name, msg)
            summary["failed"].append({"disease": name, "error": msg})
            summary["per_disease"][name] = {"status": "failed", "error": msg}
        else:
            if isinstance(result, dict) and "error" in result:
                msg = result["error"]
                logger.error("  %s failed: %s", name, msg)
                summary["failed"].append({"disease": name, "error": msg})
                summary["per_disease"][name] = {"status": "failed", "error": msg}
            else:
                cost = float(result.get("cost_usd", 0.0) or 0.0)
                n_hyp = len(result.get("therapeutic_hypotheses", []))
                summary["succeeded"].append(name)
                summary["total_cost_usd"] = round(summary["total_cost_usd"] + cost, 6)
                summary["per_disease"][name] = {
                    "status": "success",
                    "cost_usd": cost,
                    "hypothesis_count": n_hyp,
                    "file": str(_hypothesis_path(name)),
                }
                logger.info("  ✓ %s — %d hypotheses, $%.4f", name, n_hyp, cost)

        # Checkpoint after each disease so an interrupted run keeps progress.
        write_summary(summary)

        # 2-second (default) pause between API calls to avoid rate limiting.
        if i < len(todo):
            time.sleep(delay)

    summary["finished_at"] = _now()
    summary["duration_seconds"] = round(time.time() - t0, 1)
    write_summary(summary)
    return summary


def print_report(summary: dict) -> None:
    """Print a concise end-of-run report of successes and failures."""
    bar = "=" * 72
    print(f"\n{bar}")
    print(f"  HYPOTHESIS GENERATION COMPLETE — {summary['duration_seconds']}s")
    print(bar)
    print(f"  Diseases in DB : {summary['total_diseases']}")
    print(f"  Already cached : {summary['already_had']} (skipped)")
    print(f"  Attempted      : {summary['attempted']}")
    print(f"  Succeeded      : {len(summary['succeeded'])}")
    print(f"  Failed         : {len(summary['failed'])}")
    print(f"  Total API cost : ${summary['total_cost_usd']:.4f}")
    if summary["failed"]:
        print("\n  Failures:")
        for f in summary["failed"]:
            print(f"    • {f['disease']}: {f['error']}")
    print(f"\n  Summary written to {SUMMARY_PATH}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse args and run (or preview) the hypothesis backfill."""
    parser = argparse.ArgumentParser(
        description="Generate research hypotheses for every disease missing one."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview which diseases would be generated; make no API calls.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N missing diseases.")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS,
                        help=f"Seconds between API calls (default {DEFAULT_DELAY_SECONDS}).")
    parser.add_argument("--refresh", action="store_true",
                        help="Regenerate every disease even if it already has a file.")
    args = parser.parse_args()

    summary = run(dry_run=args.dry_run, limit=args.limit,
                  delay=args.delay, refresh=args.refresh)
    if not args.dry_run:
        print_report(summary)


if __name__ == "__main__":
    main()
