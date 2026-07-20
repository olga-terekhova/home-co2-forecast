# Lightweight image for a single-shot HA history fetch.
# Expected to be invoked periodically by an external scheduler
# (e.g. host cron running "docker run --rm ..."), not to loop internally.

FROM python:3.12-slim

WORKDIR /app

# requests' dependencies (urllib3, certifi, idna, charset-normalizer) are
# pure Python / ship manylinux wheels, so no build-essential / gcc is
# needed here - keeps the image small.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ingest.py predict_co2.py .
# Add further sibling modules here as they're introduced (e.g. preprocess.py).

# Run as a non-root user.
RUN useradd --create-home --shell /usr/sbin/nologin appuser
USER appuser

# Single-shot execution: the script runs once and exits.
# Env vars HA_HOSTNAME, HA_TOKEN, HA_ENTITY_ID must be supplied at
# "docker run" time (e.g. via --env / --env-file).
CMD ["python", "predict_co2.py"]