"""
Parking Occupancy Collector
============================
Polls a parking API every hour and appends only genuinely new records to a
CSV that is ready for use in prediction models.

Key behaviours
--------------
* Skips a record if its last_updated timestamp has not changed since the
  previous fetch (per parking_id).
* Handles parking meters being added or removed between fetches.
* Groups records by parking_id so partial API updates (some IDs missing)
  do not corrupt historical state for the missing ones.
* Derives extra ML-friendly columns: occupancy_rate, hour, day_of_week,
  is_weekend, date.

Usage
-----
    python parking_collector.py                        # uses defaults
    python parking_collector.py --url  <API_URL>       # override API URL
    python parking_collector.py --out  records.csv     # override output file
    python parking_collector.py --state state.json     # override state file
    python parking_collector.py --interval 3600        # seconds between polls
    python parking_collector.py --once                 # single fetch, then exit
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_API_URL  = "https://api.golemio.cz/v3/parking-measurements"   # <-- change me
DEFAULT_OUT_CSV  = "parking_records.csv"
DEFAULT_STATE    = "parking_state.json"
DEFAULT_INTERVAL = 3600   # seconds

# Columns written to the CSV (order matters for ML pipelines)
CSV_COLUMNS = [
    "parking_id",
    "primary_source",
    "primary_source_id",
    "last_updated",           # ISO-8601, UTC
    "fetched_at",             # when *we* retrieved this record
    "total_spot_number",
    "free_spot_number",
    "occupied_spot_number",
    "closed_spot_number",
    "has_free_spots",
    "occupancy_rate",         # occupied / total  (0.0–1.0, NaN-safe)
    "hour",                   # 0-23 (from last_updated)
    "day_of_week",            # 0=Monday … 6=Sunday
    "is_weekend",             # 1 if Sat/Sun else 0
    "date",                   # YYYY-MM-DD
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("parking_collector")


# ---------------------------------------------------------------------------
# State helpers  (last-seen timestamp per parking_id)
# ---------------------------------------------------------------------------

def load_state(path: str) -> dict:
    """Return {parking_id: last_updated_iso_string}."""
    p = Path(path)
    if p.exists():
        try:
            with p.open() as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not read state file %s (%s) — starting fresh.", path, exc)
    return {}


def save_state(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)   # atomic on POSIX


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def ensure_csv(path: str) -> None:
    """Create the CSV with a header row if it does not yet exist."""
    if not Path(path).exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
        log.info("Created new CSV file: %s", path)


def append_rows(path: str, rows: list[dict]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# API fetch
# ---------------------------------------------------------------------------

def fetch_api(url: str, timeout: int = 30) -> list[dict]:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "parking-collector/1.0",
            "X-Access-Token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6NTIzOSwiaWF0IjoxNzc2NTExMzI5LCJleHAiOjExNzc2NTExMzI5LCJpc3MiOiJnb2xlbWlvIiwianRpIjoiMDhjNjdmODAtZjE1ZC00ZWI3LWE1OGMtNmIwZmYzMDE1MDBkIn0.4CSf68ns0eF3C5clxtpbM8pI_K01xTvW7kyaHVIzSmk"
            },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array, got {type(data).__name__}")
        log.info("Fetched %d records from API.", len(data))
        return data
    except urllib.error.HTTPError as exc:
        log.error("HTTP %s when fetching API: %s", exc.code, exc.reason)
        raise
    except urllib.error.URLError as exc:
        log.error("Network error fetching API: %s", exc.reason)
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        log.error("Bad API response: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Record processing
# ---------------------------------------------------------------------------

def derive_features(record: dict, fetched_at: str) -> dict:
    """Add ML-friendly columns derived from existing fields."""
    total    = record.get("total_spot_number") or 0
    occupied = record.get("occupied_spot_number") or 0

    occupancy_rate = round(occupied / total, 6) if total > 0 else ""

    last_updated = record.get("last_updated", "")
    try:
        dt = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
        hour        = dt.hour
        day_of_week = dt.weekday()          # 0=Mon, 6=Sun
        is_weekend  = 1 if day_of_week >= 5 else 0
        date        = dt.date().isoformat()
    except (ValueError, AttributeError):
        hour = day_of_week = is_weekend = date = ""

    return {
        "parking_id":          record.get("parking_id", ""),
        "primary_source":      record.get("primary_source", ""),
        "primary_source_id":   record.get("primary_source_id", ""),
        "last_updated":        last_updated,
        "fetched_at":          fetched_at,
        "total_spot_number":   record.get("total_spot_number", ""),
        "free_spot_number":    record.get("free_spot_number", ""),
        "occupied_spot_number": occupied,
        "closed_spot_number":  record.get("closed_spot_number", ""),
        "has_free_spots":      record.get("has_free_spots", ""),
        "occupancy_rate":      occupancy_rate,
        "hour":                hour,
        "day_of_week":         day_of_week,
        "is_weekend":          is_weekend,
        "date":                date,
    }


def process_batch(
    records: list[dict],
    state: dict,
    fetched_at: str,
) -> tuple[list[dict], dict, dict]:
    """
    Compare incoming records against saved state.

    Returns
    -------
    new_rows   : rows to append to CSV
    new_state  : updated state dict (all IDs seen this fetch)
    summary    : {"added": int, "skipped": int, "new_ids": [...], "removed_ids": [...]}
    """
    new_rows   = []
    new_state  = dict(state)   # carry forward existing state
    seen_ids   = set()

    for rec in records:
        pid          = rec.get("parking_id")
        last_updated = rec.get("last_updated")

        if not pid:
            log.warning("Record missing parking_id — skipping: %s", rec)
            continue

        seen_ids.add(pid)
        prev_ts = state.get(pid)

        if prev_ts == last_updated:
            continue   # no change — skip

        # New or updated record
        row = derive_features(rec, fetched_at)
        new_rows.append(row)
        new_state[pid] = last_updated

    # Detect newly appearing / disappearing IDs
    prev_ids    = set(state.keys())
    new_ids     = sorted(seen_ids - prev_ids)
    removed_ids = sorted(prev_ids - seen_ids)   # missing from this fetch

    summary = {
        "added":       len(new_rows),
        "skipped":     len(records) - len(new_rows),
        "new_ids":     new_ids,
        "removed_ids": removed_ids,
    }
    return new_rows, new_state, summary


# ---------------------------------------------------------------------------
# Main poll loop
# ---------------------------------------------------------------------------

def poll_once(url: str, out_csv: str, state_path: str) -> None:
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    records = fetch_api(url)

    state = load_state(state_path)
    new_rows, new_state, summary = process_batch(records, state, fetched_at)

    if new_rows:
        ensure_csv(out_csv)
        append_rows(out_csv, new_rows)

    save_state(state_path, new_state)

    log.info(
        "Stored %d new rows, skipped %d unchanged.%s%s",
        summary["added"],
        summary["skipped"],
        f"  New IDs: {summary['new_ids']}"     if summary["new_ids"]     else "",
        f"  Removed IDs: {summary['removed_ids']}" if summary["removed_ids"] else "",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Hourly parking occupancy collector")
    parser.add_argument("--url",      default=DEFAULT_API_URL,  help="Parking API endpoint")
    parser.add_argument("--out",      default=DEFAULT_OUT_CSV,  help="Output CSV file")
    parser.add_argument("--state",    default=DEFAULT_STATE,    help="State JSON file")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, type=int,
                        help="Seconds between polls (default 3600)")
    parser.add_argument("--once",     action="store_true",
                        help="Fetch once then exit (useful for cron)")
    args = parser.parse_args()

    log.info("Parking collector starting.  URL=%s  out=%s  interval=%ds",
             args.url, args.out, args.interval)

    if args.once:
        poll_once(args.url, args.out, args.state)
        return

    while True:
        try:
            poll_once(args.url, args.out, args.state)
        except Exception as exc:
            log.error("Poll failed: %s — will retry next interval.", exc)

        log.info("Sleeping %d seconds until next poll …", args.interval)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
