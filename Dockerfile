# Use official Python slim image (matches the version the engine is tested on)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for Rasterio/GDAL
# This solves the "rasterio not installed" issues on different OSs
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p data logs

# Default to development so a container run WITHOUT credentials produces clearly
# labeled mock data (for trying it out) rather than silently failing to zeros.
# Override with -e ENVIRONMENT=production for real-data-only runs.
ENV ENVIRONMENT=development
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "main_monitor.py"]
