FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
 && rm -rf /var/lib/apt/lists/*

ENV CMAKE_BUILD_PARALLEL_LEVEL=2
ENV MAX_JOBS=2

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command for this image: run the S3 ingest script
CMD ["python", "s3_ingest.py"]
