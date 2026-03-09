FROM python:3.10-slim

WORKDIR /app

# System-Abhängigkeiten aktualisiert: libgl1 statt libgl1-mesa-glx
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Requirements kopieren und installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Den gesamten Code kopieren
COPY . .

# Falls dein Skript anders heißt, hier anpassen:
CMD ["python", "face_yolo_body.py"]