# Single image shared by both services (api + ui); the command differs per
# service in docker-compose.yml.
FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image small (no CUDA payload).
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src ./src
COPY app ./app
