# CUDA runtime + Ubuntu
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /app

# System deps + python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# (опционально) чтобы "python" работал
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]