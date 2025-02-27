# Base Image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set Working Directory
WORKDIR /app

# Copy Files
COPY . /app

# Install Dependencies
RUN pip install --upgrade pip && \
    pip install torch torchvision boto3 fastapi uvicorn pillow numpy

# Expose Port
EXPOSE 8080

# Start FastAPI App using Gunicorn
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "fastapi_app:app", "--bind", "0.0.0.0:8080"]
