FROM python:3.11-slim

# Install GLPK solver
RUN apt-get update && \
    apt-get install -y glpk-utils && \
    apt-get clean

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8080"]
