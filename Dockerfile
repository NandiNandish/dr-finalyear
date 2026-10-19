# Quantum-Enhanced Diabetic Retinopathy Screening System
# Docker Container Specification

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

# Install system dependencies required by OpenCV and scientific libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specification and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and configurations
COPY configs/ ./configs/
COPY data/ ./data/
COPY src/ ./src/
COPY app/ ./app/
COPY checkpoints/ ./checkpoints/
COPY tests/ ./tests/
COPY train.py evaluate.py ablation.py ./

# Create encrypted storage directory and generate demo dataset if needed
RUN python data/prepare_datasets.py --demo

# Expose web service port
EXPOSE 5000

# Run Flask screening application
CMD ["python", "app/app.py"]
