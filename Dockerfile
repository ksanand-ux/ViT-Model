# Base Image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime AS builder

# Set Working Directory
WORKDIR /app

# Copy Everything Except the Model
COPY . /app

# Install Dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    rm -rf /root/.cache/pip  # Free up space

# Install AWS CLI (Needed to Fetch Model from S3) - Without Prompt
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y awscli && rm -rf /var/lib/apt/lists/*

# Allow AWS CLI & SDK to Use EC2 IAM Role
ENV AWS_EC2_METADATA_DISABLED=false

# Download Model from S3 using IAM Role (NO Explicit Credentials Required)
RUN aws s3 cp s3://e-see-vit-model/models/fine_tuned_vit_imagenet100_scripted.pt /app/

# Verify Gunicorn Installation
RUN which gunicorn && gunicorn --version

# Expose Port for FastAPI
EXPOSE 8080

# Start FastAPI App using Gunicorn
CMD ["gunicorn", "fastapi_app:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080", "--workers", "1"]
