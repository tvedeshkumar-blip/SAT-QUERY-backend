FROM python:3.11-slim

# Install system dependencies for GDAL/Rasterio and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend codebase
COPY app ./app

EXPOSE 8000

ENV PORT=8000
ENV DEVICE=auto

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
