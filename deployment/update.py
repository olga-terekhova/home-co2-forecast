"""
Push a predicted value into Home Assistant as a sensor entity's state.

This is the last step of a serving cycle (see predict_co2.py): after a
prediction is made (or a cycle fails and "unknown" should be shown
instead), push_prediction_to_ha writes it to HA_OUTPUT_ENTITY_ID via
HA's REST API, creating the entity on first use.

Deliberately does not catch or log its own errors - a network error or
non-2xx response raises requests.exceptions.RequestException, and it
is the caller's job to decide what that means for the exit code (see
predict_co2.push_status). Logging here as well would just duplicate
whatever the caller already logs.
"""

import requests


def push_prediction_to_ha(hostname, token, entity_id, value, unit=None, friendly_name=None):
    """Push a calculated value to Home Assistant as an entity state.

    Creates the entity if it does not exist yet, or updates it if it does.
    State is not persisted across HA restarts (shows "unavailable" until
    the next successful push).
    """
    url = f"https://{hostname}/api/states/{entity_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    attributes = {}
    if unit:
        attributes["unit_of_measurement"] = unit
    if friendly_name:
        attributes["friendly_name"] = friendly_name

    payload = {"state": str(value), "attributes": attributes}

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()