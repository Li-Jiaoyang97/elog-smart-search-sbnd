
FROM python:3.11-slim AS builder

WORKDIR /app

#install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

#slim runtime image
FROM python:3.11-slim

WORKDIR /app

# runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    libffi8 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder step
COPY --from=builder /install /usr/local

# Copy application files
COPY ElogSemanticSearch /app/ElogSemanticSearch
# COPY certs /certs

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "ElogSemanticSearch.RESTAPI_Qdrant_HTML:app", "--host", "0.0.0.0", "--port", "8000"]
