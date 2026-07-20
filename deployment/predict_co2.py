#!/usr/bin/env python3
"""
Orchestrates a single serving cycle: fetch recent CO2 readings for one
Home Assistant entity (ingest.fetch_recent_co2), validate and prepare
the prediction window (ingest.build_prediction_window), and write the
result to CSV for the downstream serve_01/serve_02 notebooks to read.

Expects these environment variables to be set:
    HA_HOSTNAME  - hostname only, no scheme, no port (e.g. ha.example.com)
    HA_TOKEN     - Home Assistant long-lived access token
    HA_ENTITY_ID - target entity id (e.g. sensor.i_9psl_carbon_dioxide)

Optional environment variable:
    HA_OUTPUT_PATH - path to write the CSV to (default: last_co2_values.csv
                      in the current working directory). Useful for pointing
                      at a mounted volume path when run in Docker.

Exit codes:
    0 - success, or a fetch that returned no rows (not an error)
    1 - missing required env var, or the HA fetch itself failed
    2 - fetch succeeded but the prediction window could not be built
        (stale latest reading, an excessive gap, or no numeric readings
        left after filtering); see ingest.build_prediction_window
"""

import csv
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

import ingest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("predict_co2")

DEFAULT_CSV_PATH = "last_co2_values.csv"

# Optional: override where the CSV is written (e.g. a mounted volume path
# in Docker). Falls back to DEFAULT_CSV_PATH in the current working
# directory if not set.
CSV_OUTPUT_PATH = os.environ.get("HA_OUTPUT_PATH", DEFAULT_CSV_PATH)


def write_csv(readings, output_path):
    """Write readings (list of dicts with value/timestamp keys) to CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["value", "timestamp"])
        writer.writeheader()
        writer.writerows(readings)


def main():
    hostname = ingest.get_required_env("HA_HOSTNAME")
    token = ingest.get_required_env("HA_TOKEN")
    entity_id = ingest.get_required_env("HA_ENTITY_ID")

    if not hostname or not token or not entity_id:
        sys.exit(1)

    now = datetime.now(timezone.utc)
    long_window_start = now - timedelta(minutes=ingest.DEFAULT_LONG_WINDOW_MINUTES)

    readings = ingest.fetch_recent_co2(hostname, token, entity_id, long_window_start)

    if readings is None:
        # Fetch failed; error already logged. Exit non-zero so an external
        # scheduler can see the cycle failed, but do not retry here.
        sys.exit(1)

    if not readings:
        # Request succeeded but returned no rows; not an error condition.
        log.info("No rows to write for entity: %s", entity_id)
        sys.exit(0)

    window_readings = ingest.build_prediction_window(
        readings,
        now,
        ingest.DEFAULT_WINDOW_MINUTES,
        ingest.MAX_STALENESS_MINUTES,
        ingest.MAX_GAP_MINUTES,
    )

    if window_readings is None:
        # Staleness check, gap check, or no usable numeric readings;
        # reason already logged by ingest.build_prediction_window.
        # Distinct exit code from env/fetch failures so the scheduler
        # can tell "no data" apart from "stale or gappy data"
        # (PROJECT.md open issue 1).
        sys.exit(2)  # will be "unknown" state for the target entity

    write_csv(window_readings, CSV_OUTPUT_PATH)
    log.info("Saved %d rows to %s", len(window_readings), CSV_OUTPUT_PATH)


if __name__ == "__main__":
    main()