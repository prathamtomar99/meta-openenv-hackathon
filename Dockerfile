FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY environment/ ./environment/
COPY openenv.yaml .
COPY inference.py .

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start server
CMD ["python", "-m", "uvicorn", "environment.server:app", "--host", "0.0.0.0", "--port", "8000"]
