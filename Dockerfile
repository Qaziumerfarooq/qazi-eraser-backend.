FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -l -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces use port 7860 by default
ENV PORT=7860
EXPOSE 7860

CMD ["python", "server.py"]
