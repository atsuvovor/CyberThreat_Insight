# =========================================
# Base image
# =========================================
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# =========================================
# System dependencies
# Needed for ML packages and compilation
# =========================================
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# =========================================
# Copy and install Python dependencies
# =========================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =========================================
# Copy project files
# =========================================
COPY . .

# =========================================
# Environment variables (optional)
# Streamlit support
# =========================================
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV PYTHONUNBUFFERED=1

# =========================================
# Entrypoint
# =========================================
ENTRYPOINT ["python", "main.py"]
# Default pipeline stage
CMD ["--stage", "all"]

# =========================================
# Expose Streamlit port (if using dashboard)
# =========================================
EXPOSE 8501
