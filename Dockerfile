FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with specific MLflow version
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir mlflow==2.10.0 && \
    pip uninstall -y mlflow-skinny || true

# Copy application code
COPY . .

# Make serve script executable
RUN chmod +x serve_model.sh

# Expose ports
EXPOSE 5002 8501

# Default command
CMD ["python", "train.py"]

# copy and mark the script executable
COPY serve_model.sh /app/serve_model.sh
RUN chmod +x /app/serve_model.sh

# keep the default workdir
WORKDIR /app

RUN apt-get update && apt-get install -y iputils-ping && rm -rf /var/lib/apt/lists/*
