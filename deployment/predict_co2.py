#!/usr/bin/env python3
"""
Query Home Assistant history for a single entity over the last N minutes
via the REST API, and return the readings as a list of dicts (for direct
in-memory use by an ML pipeline), while also writing them to a CSV file
when run as a script.

Expects these environment variables to be set:
    HA_HOSTNAME  - hostname only, no scheme, no port (e.g. ha.example.com)
    HA_TOKEN     - Home Assistant long-lived access token
    HA_ENTITY_ID - target entity id (e.g. sensor.i_9psl_carbon_dioxide)

Optional environment variable:
    HA_OUTPUT_PATH - path to write the CSV to (default: ha_history.csv in
                      the current working directory). Useful for pointing
                      at a mounted volume path when run in Docker.

Behavior notes (carried over from the validated PowerShell version):
    - start_time is computed as UTC "now minus N minutes", formatted with
      a literal "Z" suffix (ISO 8601). Naive/local datetimes must be
      avoided here, since an incorrect UTC offset causes HA to silently
      return zero rows instead of an error.
    - minimal_response is used to reduce payload size. Only the first
      reading in the returned series has entity_id/attributes/last_updated;
      all rows have "state" and "last_changed", so last_changed is used
      as the timestamp field (see handover doc for full rationale).
    - state values are kept as strings, matching HA's raw response and the
      original PowerShell behavior. No numeric casting is done here.
    - No retry logic: on any failure, the error is logged and the function
      returns None. A single run is expected to be triggered periodically
      by an external scheduler (cron / host scheduler), so a failed cycle
      is simply skipped and picked up again on the next scheduled run.
"""

import csv
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("ha_history")

DEFAULT_HISTORY_MINUTES = 90
DEFAULT_CSV_PATH = "ha_values.csv"
REQUEST_TIMEOUT_SECONDS = 30

# Optional: override where the CSV is written (e.g. a mounted volume path
# in Docker). Falls back to DEFAULT_CSV_PATH in the current working
# directory if not set.
CSV_OUTPUT_PATH = os.environ.get("HA_OUTPUT_PATH", DEFAULT_CSV_PATH)


def get_required_env(name):
    """Read a required environment variable, or return None and log an error."""
    value = os.environ.get(name)
    if not value:
        log.error("%s environment variable is not set.", name)
        return None
    return value


def build_start_time(minutes):
    """Return a UTC ISO 8601 timestamp string with a literal Z suffix,
    representing 'now minus minutes'. Uses a timezone-aware datetime to
    avoid the local-offset bug seen with naive DateTimes in the PowerShell
    version.
    """
    start = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    return start.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_ha_history(hostname, token, entity_id, minutes=DEFAULT_HISTORY_MINUTES):
    """Fetch history for a single entity from the HA REST API.

    Returns a list of dicts with keys "value" and "timestamp", sorted
    newest-first, ready to be handed directly to an ML model. Returns
    None on any failure (network error, non-200 response, unexpected
    payload shape, empty result). Errors are logged; callers should treat
    None as "skip this cycle."
    """
    start_time = build_start_time(minutes)
    url = f"https://{hostname}/api/history/period/{start_time}"
    # NOTE: requests silently drops params whose value is None, so
    # minimal_response cannot be passed via the params dict as a bare
    # flag - it would just be omitted from the request. It is appended
    # to the URL manually instead, matching the validated PowerShell
    # request shape (?filter_entity_id=...&minimal_response with no
    # trailing "=").
    params = {
        "filter_entity_id": entity_id,
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    try:
        prepared = requests.Request(
            "GET", url, headers=headers, params=params
        ).prepare()
        prepared.url = f"{prepared.url}&minimal_response"
        session = requests.Session()
        response = session.send(prepared, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        log.error("Request to HA history API failed: %s", exc)
        return None

    try:
        payload = response.json()
    except ValueError as exc:
        log.error("Failed to parse HA response as JSON: %s", exc)
        return None

    if not payload or not isinstance(payload, list) or not payload[0]:
        log.warning("No history data returned for entity: %s", entity_id)
        return []

    # payload[0] is the array of readings for the requested entity, since
    # filter_entity_id restricts the response to a single entity.
    readings = []
    for item in payload[0]:
        state = item.get("state")
        timestamp = item.get("last_changed")
        if state is None or timestamp is None:
            log.warning("Skipping malformed reading: %s", item)
            continue
        readings.append({"value": state, "timestamp": timestamp})

    readings.sort(key=lambda r: r["timestamp"], reverse=True)
    return readings


def write_csv(readings, output_path):
    """Write readings (list of dicts with value/timestamp keys) to CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["value", "timestamp"])
        writer.writeheader()
        writer.writerows(readings)


def main():
    hostname = get_required_env("HA_HOSTNAME")
    token = get_required_env("HA_TOKEN")
    entity_id = get_required_env("HA_ENTITY_ID")

    if not hostname or not token or not entity_id:
        sys.exit(1)

    readings = fetch_ha_history(hostname, token, entity_id, DEFAULT_HISTORY_MINUTES)

    if readings is None:
        # Fetch failed; error already logged. Exit non-zero so an external
        # scheduler can see the cycle failed, but do not retry here.
        sys.exit(1)

    if not readings:
        # Request succeeded but returned no rows; not an error condition.
        log.info("No rows to write for entity: %s", entity_id)
        sys.exit(0)

    write_csv(readings, CSV_OUTPUT_PATH)
    log.info("Saved %d rows to %s", len(readings), CSV_OUTPUT_PATH)


if __name__ == "__main__":
    main()
