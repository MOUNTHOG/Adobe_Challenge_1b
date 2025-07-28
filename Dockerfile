# Dockerfile for PDF Processing Pipeline
FROM --platform=linux/amd64 python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies required for PDF processing
RUN apt-get update && apt-get install -y \
    # Required for PyMuPDF (fitz)
    libmupdf-dev \
    libfreetype6-dev \
    libjpeg-dev \
    libopenjp2-7-dev \
    libpng-dev \
    libtiff5-dev \
    zlib1g-dev \
    # Required for other dependencies
    gcc \
    g++ \
    pkg-config \
    # Clean up to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY all.py .
COPY final.py .
COPY src/ ./src/

# Create necessary directories
RUN mkdir -p collections headings

# Copy input data if available (optional - can be mounted as volume)
# COPY collections/ ./collections/

# Set permissions
RUN chmod +x final.py main.py all.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; import src.pdf_classifier; print('Health check passed')" || exit 1

# Default command
CMD ["python", "final.py"]