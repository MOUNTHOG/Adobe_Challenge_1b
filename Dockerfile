# Use slim Python base image targeting linux/amd64 platform
FROM --platform=linux/amd64 python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies for PDF & image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OCR with Tesseract
    tesseract-ocr \
    tesseract-ocr-eng \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # PDF rendering utilities
    poppler-utils \
    # Image/PDF processing & build tools
    libmupdf-dev \
    libfreetype6-dev \
    libjpeg-dev \
    libopenjp2-7-dev \
    libpng-dev \
    libtiff5-dev \
    zlib1g-dev \
    gcc \
    g++ \
    pkg-config \
    # Clean up
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py all.py final.py ./
COPY src/ ./src/

# Create necessary directories
RUN mkdir -p collections headings

# Set execution permissions
RUN chmod +x final.py main.py all.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src.pdf_classifier; print('Health check passed')" || exit 1

# Default command
CMD ["python", "final.py"]
