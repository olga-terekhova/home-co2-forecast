#!/usr/bin/env python3
"""
Orchestrates a single serving cycle: fetch recent CO2 readings for one
Home Assistant entity (ingest.fetch_recent_co2), validate and prepare
the prediction window (ingest.build_prediction_window), write the CSV,
engineer features and predict (process.run_prediction), and push the
result back into Home Assistant as a sensor entity (update.py). On
every exit path except a missing/invalid env var, an "unknown" or the
predicted value is pushed to HA_OUTPUT_ENTITY_ID so it reflects the
current state rather than silently going stale.

Expects these environment variables to be set:
    HA_HOSTNAME  - hostname only, no scheme, no port (e.g. ha.example.com)
    HA_TOKEN     - Home Assistant long-lived access token
    HA_INPUT_ENTITY_ID - input entity id (e.g. sensor.i_9psl_carbon_dioxide)
    HA_OUTPUT_ENTITY_ID - output entity id (e.g. sensor.co2_predicted_10min )

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
    3 - window built and CSV written, but the prediction stage failed:
        model file missing, feature row had NaNs (should be
        unreachable given the window checks above, but kept as a
        safety net), or a feature_names mismatch against the model;
        see process.run_prediction
    4 - everything up to and including the prediction succeeded, but
        pushing the result (or "unknown", on an earlier failure path)
        back into Home Assistant failed - e.g. HA unreachable, bad
        token, or a non-2xx response; see update.push_prediction_to_ha
"""

import csv
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

import requests

import ingest
import process
import update

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

OUTPUT_UNIT = "ppm"
OUTPUT_FRIENDLY_NAME = "CO2 Predicted (10 min)"


def write_csv(readings, output_path):
    """Write readings (list of dicts with value/timestamp keys) to CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["value", "timestamp"])
        writer.writeheader()
        writer.writerows(readings)


def push_status(hostname, token, entity_id, value):
    """Push value to HA_OUTPUT_ENTITY_ID, logging (not raising) on failure.

    A network error or non-2xx response from HA must not crash the
    script - that would replace whatever exit code/log message the
    caller was about to produce with an opaque traceback. Returns True
    on success, False on failure (already logged); callers on the
    early-exit paths can ignore a False return (the original failure
    already has its own exit code), but the success path below treats
    a False return as its own failure (exit code 4).
    """
    try:
        update.push_prediction_to_ha(
            hostname=hostname,
            token=token,
            entity_id=entity_id,
            value=value,
            unit=OUTPUT_UNIT,
            friendly_name=OUTPUT_FRIENDLY_NAME,
        )
        return True
    except requests.exceptions.RequestException as exc:
        log.error(
            "Failed to push %r to Home Assistant entity %s: %s", value, entity_id, exc
        )
        return False


def main():
    hostname = ingest.get_required_env("HA_HOSTNAME")
    token = ingest.get_required_env("HA_TOKEN")
    input_entity_id = ingest.get_required_env("HA_INPUT_ENTITY_ID")
    output_entity_id = ingest.get_required_env("HA_OUTPUT_ENTITY_ID")

    if not hostname or not token or not input_entity_id or not output_entity_id:
        sys.exit(1)

    now = datetime.now(timezone.utc)
    long_window_start = now - timedelta(minutes=ingest.DEFAULT_LONG_WINDOW_MINUTES)

    readings = ingest.fetch_recent_co2(hostname, token, input_entity_id, long_window_start)

    if readings is None:
        # Fetch failed; error already logged. Exit non-zero so an external
        # scheduler can see the cycle failed, but do not retry here.
        # A failed push here is logged but does not change the exit
        # code below - the fetch failure is the primary problem.
        push_status(hostname, token, output_entity_id, "unknown")
        sys.exit(1)

    if not readings:
        # Request succeeded but returned no rows; not an error condition.
        log.info("No rows to write for entity: %s", input_entity_id)
        push_status(hostname, token, output_entity_id, "unknown")
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
        push_status(hostname, token, output_entity_id, "unknown")
        sys.exit(2)  # will be "unknown" state for the target entity

    write_csv(window_readings, CSV_OUTPUT_PATH)
    log.info("Saved %d rows to %s", len(window_readings), CSV_OUTPUT_PATH)

    try:
        value, based_on = process.run_prediction(
            CSV_OUTPUT_PATH, process.MODEL_PATH, process.PREDICTION_OUTPUT_PATH
        )
    except FileNotFoundError as exc:
        log.error("%s", exc)
        push_status(hostname, token, output_entity_id, "unknown")
        sys.exit(3)
    except process.PredictionError as exc:
        log.error("%s", exc)
        push_status(hostname, token, output_entity_id, "unknown")
        sys.exit(3)

    log.info(
        "Predicted %.2f ppm for %s (based on %s); wrote to %s",
        value,
        based_on + timedelta(minutes=process.HORIZON_MINUTES),
        based_on,
        process.PREDICTION_OUTPUT_PATH,
    )

    # Here, unlike the earlier branches, the push IS the last remaining
    # step - if it fails, the whole cycle should be reported as failed
    # even though the prediction itself succeeded, or a scheduler
    # watching only the exit code would believe HA was updated when it
    # was not.
    if not push_status(hostname, token, output_entity_id, value):
        sys.exit(4)
    
    import resource

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with open("/app/out/mem_usage.log", "a") as f:
        f.write(f"{peak_kb / 1024:.1f} MB\n")


if __name__ == "__main__":
    main()