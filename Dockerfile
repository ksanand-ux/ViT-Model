# Base Image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set Working Directory
WORKDIR /app

# Copy Files
COPY . /app

# Install Dependencies from requirements.txt
# Single Layer Installation to Avoid Cache Issues
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Verify Gunicorn Installation
RUN which gunicorn && gunicorn --version

# Expose Port for FastAPI
EXPOSE 8080

# Start FastAPI App using Gunicorn with Optimized Workers
CMD ["gunicorn", "fastapi_app:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080", "--workers", "4"]
