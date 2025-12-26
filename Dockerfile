FROM python:3.11-slim

# Install system dependencies + GLPK solver
RUN apt-get update && apt-get install -y --no-install-recommends \
    glpk-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency file
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Railway provides $PORT environment var
ENV PORT=8000

# Start FastAPI server
CMD ["sh", "-c", "uvicorn backend:app --host 0.0.0.0 --port $PORT"]
