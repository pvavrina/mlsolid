# Use a slim Python image for a smaller footprint
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependency file and install packages
# The --no-cache-dir flag optimizes space
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code
COPY app/ app/

# Command to start the Uvicorn/FastAPI application
# The port 8000 is exposed (K8s Service will map this)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
