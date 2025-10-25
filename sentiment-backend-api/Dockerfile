# Use the official Python base image
FROM python:3.10-slim

# Set environment variables for non-LLM deployment
ENV PYTHONUNBUFFERED True

# Set the working directory inside the container
WORKDIR /app

# Copy dependency files and install Python packages
COPY requirements.txt .
# Install gunicorn for production-grade serving
RUN pip install --no-cache-dir gunicorn
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Cloud Run injects the PORT environment variable. We use gunicorn to run Flask.
# Use 0.0.0.0 to listen on all available IPs.
EXPOSE 8080

# Command to run the application using Gunicorn, linking the app instance in main.py
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app