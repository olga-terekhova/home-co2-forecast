"""
Fetch CO2 history for a single Home Assistant entity via the REST API,
and turn it into a minimally viable dataset for prediction: a list of
{"value", "timestamp"} dicts covering exactly the prediction window
(latest reading plus 60 minutes back), with staleness and gap checks
applied and the left edge filled in where needed.

No pandas here by design - this module's output contract is a plain
list of raw-string readings, ready for preprocess.py (or serve_01) to
turn into a DataFrame and engineer features from.

Behavior notes (carried over from the validated PowerShell version):
    - start_time is computed as UTC "now minus N minutes", formatted with
      a literal "Z" suffix (ISO 8601). Naive/local datetimes must be
      avoided here, since an incorrect UTC offset causes HA to silently
      return zero rows instead of an error.
    - minimal_response is used to reduce payload size. Only the first
      reading in the returned series has entity_id/attributes/last_updated;
      all rows have "state" and "last_changed", so last_changed is used
      as the timestamp field (see handover doc for full rationale).
    - state values are kept as strings out of fetch_recent_co2, matching
      HA's raw response and the original PowerShell behavior. Numeric
      casting happens later, inside build_prediction_window.
    - No retry logic in fetch_recent_co2: on any failure, the error is
      logged and the function returns None. A single run is expected to
      be triggered periodically by an external scheduler (cron / host
      scheduler), so a failed cycle is simply skipped and picked up
      again on the next scheduled run.
"""

import logging
import os
from datetime import datetime, timedelta

import requests

log = logging.getLogger(__name__)

DEFAULT_WINDOW_MINUTES = 60
DEFAULT_LONG_WINDOW_MINUTES = 90

# Freshness of the latest fetched reading, vs. now. Independent of
# MAX_GAP_MINUTES below (which concerns spacing *within* the series).
MAX_STALENESS_MINUTES = 5

# Largest gap between two consecutive readings (or between window_start
# and the first reading) that is still treated as HA compression rather
# than an outage. Above this, the series is rejected outright.
MAX_GAP_MINUTES = 10

REQUEST_TIMEOUT_SECONDS = 30


def get_required_env(name):
    """Read a required environment variable, or return None and log an error."""
    value = os.environ.get(name)
    if not value:
        log.error("%s environment variable is not set.", name)
        return None
    return value


def fetch_recent_co2(hostname, token, entity_id, long_window_start):
    """Fetch history for a single entity from the HA REST API.

    Returns a list of dicts with keys "value" and "timestamp", sorted
    oldest-first (chronological), ready to be handed directly to an ML
    pipeline. Returns
    an empty list when the request succeeds but yields no rows, and
    None on any failure (network error, non-200 response, unexpected
    payload shape). Errors are logged; callers should treat None as
    "skip this cycle."
    """

    # Return a UTC ISO 8601 timestamp string with a literal Z suffix
    # Use a timezone-aware datetime
    start_time = long_window_start.strftime("%Y-%m-%dT%H:%M:%SZ")

    url = f"https://{hostname}/api/history/period/{start_time}"
    # NOTE: requests silently drops params whose value is None, so
    # minimal_response cannot be passed via the params dict as a bare
    # flag - it would just be omitted from the request. It is appended
    # to the URL manually instead, matching the validated
    # request (?filter_entity_id=...&minimal_response with no
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
        log.info(f"Requesting {prepared.url}")
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

    readings.sort(key=lambda r: r["timestamp"])
    return readings


def filter_numeric_readings(readings):
    """Drop readings whose value cannot be parsed as a number.

    HA can report a state of "unknown" or "unavailable" instead of a
    numeric reading. These carry no usable signal, so they are dropped
    up front, before staleness/gap/interpolation logic runs. A dropped
    reading effectively widens whatever gap it sat inside, so it is
    still subject to the same MAX_GAP_MINUTES check as a true outage -
    which is the intended behavior, not a side effect to work around.
    """
    numeric = []
    dropped = 0
    for r in readings:
        try:
            float(r["value"])
            numeric.append(r)
        except (TypeError, ValueError):
            dropped += 1
    if dropped:
        log.warning(
            "Dropped %d non-numeric reading(s) (e.g. unknown/unavailable).", dropped
        )
    return numeric


def parse_timestamp(ts_str):
    """Parse an ISO 8601 timestamp string into a timezone-aware datetime."""
    return datetime.fromisoformat(ts_str)


def get_latest_timestamp(readings):
    """Return the latest (max) timestamp among readings, as a datetime."""
    return max(parse_timestamp(r["timestamp"]) for r in readings)


def is_stale(latest, now, max_staleness_minutes):
    """Return True if latest is older than max_staleness_minutes relative to now."""
    age = now - latest
    if age > timedelta(minutes=max_staleness_minutes):
        log.warning(
            "Latest reading is %s old; exceeds staleness threshold of %d minutes.",
            age, max_staleness_minutes,
        )
        return True
    return False


def split_window(readings, latest, window_minutes):
    """Split readings into (window_start, readings_window, readings_history).

    window_start is latest floored to the minute, minus window_minutes.
    readings_window holds rows with timestamp in [window_start, latest],
    sorted ascending. readings_history holds rows strictly before
    window_start, sorted ascending. Each row keeps its original string
    "value"/"timestamp" fields; a parsed datetime is attached under "ts"
    for use by the functions below.
    """
    window_start = latest.replace(second=0, microsecond=0) - timedelta(
        minutes=window_minutes
    )

    parsed = [{**r, "ts": parse_timestamp(r["timestamp"])} for r in readings]
    parsed.sort(key=lambda r: r["ts"])

    readings_window = [r for r in parsed if r["ts"] >= window_start]
    readings_history = [r for r in parsed if r["ts"] < window_start]
    return window_start, readings_window, readings_history


def has_excessive_gap(window_start, readings_window, max_gap_minutes):
    """Check whether any gap in the window exceeds max_gap_minutes.

    Compares window_start to the first item of readings_window, then
    walks each consecutive pair within readings_window, applying one
    uniform threshold throughout. window_start is used only as a time
    anchor here; it has no associated value.
    """
    max_gap = timedelta(minutes=max_gap_minutes)
    boundary = [window_start] + [r["ts"] for r in readings_window]
    for prev_ts, curr_ts in zip(boundary, boundary[1:]):
        gap = curr_ts - prev_ts
        if gap > max_gap:
            log.warning(
                "Gap of %s between %s and %s exceeds max gap of %d minutes.",
                gap, prev_ts, curr_ts, max_gap_minutes,
            )
            return True
    return False


def interpolate_left_edge(window_start, readings_history, readings_window):
    """Build the reading at window_start if it is not already covered.

    Returns a reading dict ({"value": ..., "timestamp": ...}) to prepend
    to readings_window, or None if the first item of readings_window
    already falls in window_start's minute bucket.

    Uses time-weighted linear interpolation between the last history
    point and the first window point when history is available; falls
    back to a flat copy of the first window point when there is no
    history to interpolate from (e.g. start of sensor history).

    Values are cast to float here; by this point readings have already
    passed through filter_numeric_readings, so this is not expected to
    raise.
    """
    first = readings_window[0]
    if first["ts"].replace(second=0, microsecond=0) == window_start:
        return None

    if readings_history:
        prev = readings_history[-1]
        t_prev, t_next = prev["ts"], first["ts"]
        v_prev, v_next = float(prev["value"]), float(first["value"])
        frac = (window_start - t_prev) / (t_next - t_prev)
        value = v_prev + frac * (v_next - v_prev)
    else:
        value = float(first["value"])

    return {"value": str(value), "timestamp": window_start.isoformat()}


def build_prediction_window(
    readings, now, window_minutes, max_staleness_minutes, max_gap_minutes
):
    """Prepare the list of readings for the prediction window.

    Drops non-numeric readings, runs the staleness check, splits into
    window/history, runs the uniform gap check (window_start boundary
    plus interior gaps), and synthesizes the left-edge reading if
    needed.

    Returns the final list of readings (ascending, raw timestamps,
    except a synthesized left-edge row which carries window_start), or
    None if any check fails and the cycle should be aborted.
    """
    readings = filter_numeric_readings(readings)
    if not readings:
        log.warning("No numeric readings available after filtering.")
        return None

    latest = get_latest_timestamp(readings)
    if is_stale(latest, now, max_staleness_minutes):
        return None

    window_start, readings_window, readings_history = split_window(
        readings, latest, window_minutes
    )

    if not readings_window:
        # Defensive only: latest itself always satisfies ts >= window_start,
        # so readings_window should never be empty here.
        log.warning("No readings found in the prediction window.")
        return None

    if has_excessive_gap(window_start, readings_window, max_gap_minutes):
        return None

    edge = interpolate_left_edge(window_start, readings_history, readings_window)

    result = [{"value": r["value"], "timestamp": r["timestamp"]} for r in readings_window]
    if edge is not None:
        result.insert(0, edge)

    return result