# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Default entry point
ENTRYPOINT ["python", "main.py"]

# Default command (can be overridden)
CMD ["--stage", "all"]
