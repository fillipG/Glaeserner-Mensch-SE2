# 1. Basis-Image: Schlank und stabil
FROM python:3.10-slim

# 2. Umgebungsvariablen setzen
# Verhindert, dass Python .pyc-Dateien schreibt und puffert (bessere Logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. System-Abhängigkeiten (Aktualisiert für Debian Trixie/Bookworm)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. Arbeitsverzeichnis
WORKDIR /app

# 5. Abhängigkeiten installieren
# Nutzt den Docker-Cache: Nur wenn sich die requirements.txt ändert, wird neu installiert
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Projektdateien kopieren
# Body-YOLO: yolov8n.pt wird fuer die Person-Erkennung benoetigt.
# Dieses Modell wird nur geladen wenn moondream_crop_mode in config.yaml
# auf "body" oder "shadow" gesetzt ist.
# Wenn Body-YOLO nicht benoetigt wird oder Performance-Probleme auftreten:
# 1. moondream_crop_mode: face in config.yaml setzen
# 2. oder diese Zeile auskommentieren und den Container neu bauen:
COPY yolov8n.pt .
# Segmentation-Modell fuer body_seg Modus
# Um body_seg zu deaktivieren: diese Zeile auskommentieren und Container neu bauen
COPY yolov8n-seg.pt .
COPY . .

# 8. Startbefehl
CMD ["python", "face_yolo.py"]
