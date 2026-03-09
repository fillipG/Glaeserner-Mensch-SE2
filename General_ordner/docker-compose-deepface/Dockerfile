# Nutzung der NVIDIA-Basis Unterstützung
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Installation von Python 3.10 und System-Bibliotheken für OpenCV
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python-Link erstellen (Standard Python 3.10)
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Pip aktualisieren
RUN python -m pip install --upgrade pip

WORKDIR /app

# Anforderungen kopieren und installieren
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Skript kopieren
COPY script.py .

# Verzeichnisse für die Pipeline (Laut Protokoll)
RUN mkdir -p /data/input /data/processed

CMD ["python", "script.py"]