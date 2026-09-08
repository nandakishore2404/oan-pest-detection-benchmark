# ==============================================================================
# OAN Kenya Pest Detection Model Benchmarking Lab - Container Image
# ==============================================================================

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies (OpenCV, libGL, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU or CUDA
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set permissions
RUN chmod -R 755 /app

# Expose default API microservice port
EXPOSE 8000

# Default entrypoint: Run Milestone 1 benchmark on sample image
CMD ["python", "scripts/benchmark.py", "--image", "data/golden/maize_fall_armyworm_01.jpg", "--models", "all"]
