import time
import os
import re
import yaml
from deepface import DeepFace

"""
Anleitung zur Ausführung im Docker-Container:

1) Ordnerstruktur sicherstellen:
    ./faces_yolo (Input) und ./final (Output)
2) Docker-Image bauen:
    docker build -t deepface_ai .
3) Container starten:
    Das Skript läuft automatisch und überwacht den Eingabeordner.
"""

# Pfade innerhalb des Docker-Containers
INPUT_DIR = "/data/input"
PROCESSED_DIR = "/data/processed"
CONFIG_PATH = "/app/config.yaml"

# Output-Dir anlegen, falls noch nicht vorhanden
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_config():
    """Liest die DeepFace-Settings aus der config.yaml."""
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
            for model in cfg.get("pipeline", []):
                if model.get("id") == "deepface":
                    return model
    except Exception as e:
        print(f"Fehler beim Laden der Config: {e}")

    # Fallback-Werte
    return {"enabled": True, "Deepface_emotion": True, "Deepface_alter": True, "Deepface_geschlecht": True}


print(f"DeepFace Worker aktiv. Überwache: {INPUT_DIR}")

while True:
    # 1. Aktuelle Konfiguration prüfen
    config = load_config()
    if not config.get("enabled", True):
        time.sleep(5)
        continue

    # 2. Eingangsordner nach Bildern scannen
    try:
        files = os.listdir(INPUT_DIR)
        valid_files = [f for f in files if re.match(r'^face_?\d+', f, re.IGNORECASE)]
    except Exception as e:
        print(f"Fehler beim Scan: {e}")
        time.sleep(2)
        continue

    # 3. Dateien nacheinander verarbeiten
    for filename in valid_files:
        img_path = os.path.join(INPUT_DIR, filename)
        face_id = os.path.splitext(filename)[0]

        yaml_path = os.path.join(PROCESSED_DIR, f"{face_id}_deepface.yaml")
        sketch_path = os.path.join(PROCESSED_DIR, f"{face_id}_sketch.jpg")

        if os.path.exists(yaml_path):
            continue

        try:
            print(f"Analysiere {filename}...")

            # detector_backend='skip', da YOLO den Crop bereits erledigt hat
            results = DeepFace.analyze(
                img_path=img_path,
                actions=["age", "gender", "emotion"],
                enforce_detection=False,
                detector_backend='skip',
                silent=True
            )
            res = results[0] if isinstance(results, list) else results

            # Daten zusammenbauen
            out = {}
            if config.get("Deepface_emotion", True):
                out["Emotion"] = res.get("dominant_emotion")
            if config.get("Deepface_alter", True):
                out["Alter"] = int(res.get("age", 0))
            if config.get("Deepface_geschlecht", True):
                out["Geschlecht"] = "Mann" if res.get("dominant_gender") == "Man" else "Frau"

            out["Confidence"] = round(float(res.get("face_confidence", 0)), 4)

            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(out, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                f.flush()
                os.fsync(f.fileno())


            print(f"Gespeichert: {os.path.basename(yaml_path)}")

        except Exception as e:
            print(f"DeepFace Fehler bei {filename}: {e}")

    # 4. Kurze Pause vor dem nächsten Scan
    time.sleep(1)