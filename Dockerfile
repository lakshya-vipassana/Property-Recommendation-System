# =====================================================
# Base image: Python for amd64 (x86_64)
# =====================================================
FROM --platform=linux/amd64 python:3.9-slim

# =====================================================
# Environment settings
# =====================================================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# =====================================================
# Set working directory
# =====================================================
WORKDIR /app

# =====================================================
# Install system dependencies (minimal)
# =====================================================
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# =====================================================
# Copy requirements first (better caching)
# =====================================================
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# =====================================================
# Copy application code
# =====================================================
COPY . .

# =====================================================
# Expose Streamlit port
# =====================================================
EXPOSE 8501

# =====================================================
# Run Streamlit app (presentation/demo)
# =====================================================
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
