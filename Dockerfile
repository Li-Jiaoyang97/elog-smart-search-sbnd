
FROM python:3.11.4-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


FROM python:3.11.4-slim
WORKDIR /app

# Copy installed dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11


COPY ElogSemanticSearch /app/ElogSemanticSearch
#COPY certs /certs
COPY .env /app/.env


EXPOSE 8000
#EXPOSE 443

#CMD ["python", "-m", "uvicorn", "ElogSemanticSearch.RESTAPI_Qdrant_HTML:app", "--host", "0.0.0.0", "--port", "443", "--ssl-keyfile=/certs/key.pem", "--ssl-certfile=/certs/cert.pem"]


CMD ["python", "-m", "uvicorn", "ElogSemanticSearch.RESTAPI_Qdrant_HTML:app", "--host", "0.0.0.0", "--port", "8000"]

#CMD ["uvicorn", "ElogSemanticSearch.RESTAPI_Qdrant_HTML:app", "--host", "0.0.0.0", "--port", "8000"]
