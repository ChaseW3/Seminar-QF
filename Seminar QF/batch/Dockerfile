# Use a lightweight Python image
FROM python:3.9-slim

# Install system dependencies (needed for compilation or libraries)
# build-essential for compiling C extensions if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
# We include google-cloud-storage manually
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir google-cloud-storage pandas-gbq

# Copy the source code
# We copy the 'src' directory to /app/src
COPY src/ /app/src/

# Copy the batch driver
COPY batch/batch_driver.py /app/batch_driver.py

# Set PYTHONPATH to include the project root
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Metadata
LABEL description="Monte Carlo GARCH Batch Worker"

# Entrypoint
ENTRYPOINT ["python", "/app/batch_driver.py"]
