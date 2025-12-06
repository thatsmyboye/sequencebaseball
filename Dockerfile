# Dockerfile for Railway deployment - Optimized
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
# Use lite mode for lower memory
ENV API_MODE=lite

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (for better caching)
COPY requirements-railway.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-railway.txt

# Copy application code
COPY api/ ./api/
COPY pitch_viz.py sequence_visualizations.py ./
COPY start.py run_api.py ./

# Copy data directory (contains .gitkeep at minimum)
COPY data/ ./data/

# Ensure cache directory exists
RUN mkdir -p /app/data/cache

# Expose port
EXPOSE 8000

# Run the startup script
CMD ["python", "start.py"]
