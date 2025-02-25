# Base Image for FastAPI with PyTorch Support
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set Working Directory
WORKDIR /app

# Copy Files
COPY . /app

# Install Dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Port
EXPOSE 8080

# Run FastAPI App
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8080"]
