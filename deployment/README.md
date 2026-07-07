# HA history fetch

Single-shot script that queries Home Assistant's `/api/history/period`
endpoint for one entity's last 60 minutes of readings, writes them to
`ha_history.csv`, and (when imported rather than run directly) returns them
as a list of `{"value": ..., "timestamp": ...}` dicts for direct use by an
ML pipeline.

## Environment variables (required)

- `HA_HOSTNAME` - hostname only, no scheme, no port
- `HA_TOKEN` - HA long-lived access token
- `HA_ENTITY_ID` - target entity id

## Run locally

```
pip install -r requirements.txt
export HA_HOSTNAME=ha.example.com
export HA_TOKEN=xxxx
export HA_ENTITY_ID=sensor.i_9psl_carbon_dioxide
python ha_history.py
```

## Run in Docker via docker compose (single invocation, meant for an external scheduler)

```
cp .env.example .env
# edit .env with real values

docker compose run --rm ha-history
```

This is single-shot by design (`restart: "no"`, no `docker compose up -d`
usage) - an external scheduler (cron, host-level scheduler, etc.) is
expected to trigger `docker compose run --rm ha-history` on its own
interval.

`.env` is picked up automatically by `docker compose` and substituted into
`docker-compose.yml`. Do not commit the filled-in `.env` file - only
`.env.example` is meant to be checked in.

The compose file mounts `./out` on the host to `/app/out` in the
container and sets `HA_OUTPUT_PATH=/app/out/ha_history.csv`, so the CSV
persists on the host after the container exits.

### Run in plain Docker (without compose)

```
docker build -t ha-history .
docker run --rm \
  -e HA_HOSTNAME=ha.example.com \
  -e HA_TOKEN=xxxx \
  -e HA_ENTITY_ID=sensor.i_9psl_carbon_dioxide \
  -e HA_OUTPUT_PATH=/app/out/ha_history.csv \
  -v "$(pwd)/out:/app/out" \
  ha-history
```

## Use as a library, in-memory

```python
from ha_history import fetch_ha_history

readings = fetch_ha_history(hostname, token, entity_id, minutes=60)
if readings is None:
    # fetch failed, error already logged - skip this cycle
    ...
else:
    # hand `readings` directly to the model
    ...
```

## Behavior carried over from the validated PowerShell version

- `filter_entity_id` and `minimal_response` (bare flag) are both used.
- `last_changed` is used as the timestamp field (present on every row,
  unlike `last_updated`, which `minimal_response` drops after the first
  row - see project handover doc for full rationale).
- `state` values are kept as strings, not cast to numeric types.
- No retry logic: a failed request is logged and the cycle is skipped;
  the next scheduled run will pick up fresh data.

## Still open (not addressed here, per the handover doc)

- Handling of `unavailable` / `unknown` state strings is not filtered or
  special-cased - they will currently pass through as literal strings in
  the output. Decide downstream (in the ML pipeline) whether to filter or
  handle these before they reach the model.
- Scheduling mechanism itself (cron, host-level scheduler, etc.) is
  external to this container, per your last message - nothing in this
  image loops or sleeps internally.