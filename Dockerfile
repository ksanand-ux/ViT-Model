# Base Image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime AS builder

# Set Working Directory
WORKDIR /app

# Copy Everything Except the Model First
COPY . /app

# Install Dependencies from requirements.txt
# Single Layer Installation to Avoid Cache Issues
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install AWS CLI (Needed to Fetch Model from S3)
RUN apt-get update && apt-get install -y awscli

# Only Change: Download Model from S3 Instead of Copying It
RUN aws s3 cp s3://e-see-vit-model/models/fine_tuned_vit_imagenet100_scripted.pt /app/

# Verify Gunicorn Installation
RUN which gunicorn && gunicorn --version

# Expose Port for FastAPI
EXPOSE 8080

# Start FastAPI App using Gunicorn with 1 Worker (For Testing CI/CD)
CMD ["gunicorn", "fastapi_app:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080", "--workers", "1"]
