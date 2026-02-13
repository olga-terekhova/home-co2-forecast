# Use Python Alpine for a lightweight image
FROM python:3.11-alpine

# Set working directory
WORKDIR /app

# Install build dependencies needed for Pandas
RUN apk add --no-cache --virtual .build-deps \
    gcc \
    g++ \
    musl-dev \
    linux-headers \
    && pip install --no-cache-dir pandas \
    && apk del .build-deps

# Copy your application files (if any)
# COPY . .

# Default command
CMD ["python"]