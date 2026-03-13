import os
# =========================================================
# KONFIGURATIONSDATEI (DOCKER-OPTIMIERT)
# =========================================================

# Im Docker-Container ist das Arbeitsverzeichnis /app
BASE_DIR = os.getcwd() 

# Input-Ordner: Hier liegen die Bilder von YOLO
# Docker-Mapping: ../faces_yolo -> /app/faces_yolo
SCAN_FOLDER = os.path.join(BASE_DIR, "faces_yolo")

# Output-Ordner: Hier landen die Ergebnisse
# Docker-Mapping: ../final -> /app/final
FINAL_FOLDER = os.path.join(BASE_DIR, "final")

# Scan-Intervall (Sekunden)
SCAN_INTERVAL = 0.5

# Unterstützte Bildformate
SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png")