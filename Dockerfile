# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create data directory
RUN mkdir -p ./data

# Copy CSV files - app.py handles multiple path locations
# Copy data directory (contains the CSV file)
COPY data/ ./data/

# Create directory for Firebase key (optional)
RUN mkdir -p /app/keys

# Create directory for saved models (persisted across restarts on some platforms)
RUN mkdir -p /app/models/categories

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application with Gunicorn for production
# Use 1 worker with 1 thread to minimize memory usage, timeout 120s, restart workers after 50 requests to prevent memory leaks
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "--worker-class", "sync", "--threads", "1", "--max-requests", "50", "--max-requests-jitter", "10", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
