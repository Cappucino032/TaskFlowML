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
# Copy data directory if it exists in the build context
COPY data/ ./data/

# Copy CSV file from root if it exists
COPY "Survey Questions for ML Training Data.csv" ./

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
# Use 2 workers for better performance, bind to 0.0.0.0, timeout 120s for cold starts
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
