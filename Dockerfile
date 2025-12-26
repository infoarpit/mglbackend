FROM python:3.11-slim

# Install GLPK solver
RUN apt-get update && apt-get install -y glpk-utils

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Run FastAPI server
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "10000"]
