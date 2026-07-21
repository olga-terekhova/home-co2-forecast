# Image for a single-shot HA fetch, feature build, CO2 prediction, and
# push of the result back into HA as a sensor entity.
# Expected to be invoked periodically by an external scheduler
# (e.g. host cron running "docker run --rm ..."), not to loop internally.

FROM python:3.12-slim

WORKDIR /app

# NOTE: pandas/numpy/scikit-learn (needed by process.py) ship manylinux
# wheels for this base image, so build-essential/gcc still is not
# needed - but the image is no longer as lightweight as when it only
# did the HA fetch (requests' pure-Python dependency set).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ingest.py predict_co2.py process.py update.py model-regressor.pkl .
# Add further sibling modules here as they're introduced.

# Run as a non-root user.
RUN useradd --create-home --shell /usr/sbin/nologin appuser
USER appuser

# Single-shot execution: the script runs once and exits.
# Env vars HA_HOSTNAME, HA_TOKEN, HA_INPUT_ENTITY_ID, HA_OUTPUT_ENTITY_ID
# must be supplied at "docker run" time (e.g. via --env / --env-file).
CMD ["python", "predict_co2.py"]