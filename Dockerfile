# Base image — Python 3.12 slim (lightweight)
FROM python:3.12-slim

# Working directory inside container
WORKDIR /app

# System dependency — libomp for LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker cache optimization)
# If requirements don't change, this layer is cached
COPY api/requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY api/ .

# Expose port 8000
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]