# Use Python base image
FROM python:3.10-slim
WORKDIR /app
COPY . /app
# Install system dependencies (for transformers + torch)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Expose port 8080 (Cloud Run requirement)
EXPOSE 8080
# Set environment variable for Flask
ENV PORT=8080
# Run the app
CMD ["python", "main.py"]
