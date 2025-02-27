# === Stage 1: Build Dependencies ===
# Base Image for Building
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime as builder

# Set Working Directory
WORKDIR /app

# Copy Files
COPY . /app

# Install Dependencies from requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# === Stage 2: Final Image ===
# Base Image for Production
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set Working Directory
WORKDIR /app

# Copy Files and Dependencies from Builder
COPY --from=builder /app /app

# Expose Port for FastAPI
EXPOSE 8080

# Start FastAPI App using Gunicorn with Optimized Workers
CMD ["gunicorn", "fastapi_app:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080", "--workers", "4"]
