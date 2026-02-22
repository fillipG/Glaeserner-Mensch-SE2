import os

# =========================================================
# KONFIGURATIONSDATEI
# =========================================================
# Enthält alle veränderbaren Parameter für das Projekt.
# =========================================================

# Projektbasis-Verzeichnis
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input-Ordner mit Bildern
SCAN_FOLDER = os.path.join(BASE_DIR, "main_image")

# Output-Ordner für YAML-Dateien
FINAL_FOLDER = os.path.join(BASE_DIR, "final")

# Scan-Intervall (sekunden) – relevant, falls Ordner regelmäßig gescannt wird
SCAN_INTERVAL = 0.5

# Unterstützte Bildformate
SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png")