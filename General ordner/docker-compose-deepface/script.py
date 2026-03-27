import os
import time

import yaml
from deepface import DeepFace

"""
Anleitung zur Ausfuhrung im Docker-Container:

1) Ordnerstruktur sicherstellen:
    ./faces_yolo (Input) und ./final (Output)
2) Docker-Image bauen:
    docker build -t deepface_ai .
3) Container starten:
    Das Skript lauft automatisch und uberwacht den Eingabeordner.
"""

INPUT_DIR = "/app/deepface_inbox"
PROCESSED_DIR = "/app/final"
CONFIG_PATH = "/app/config.yaml"
# Schnellschalter fuer den DeepFace-Detector:
# True  -> bessere Robustheit/Qualitaet (langsamer)
# False -> YOLO-Crops direkt nutzen (schneller)
USE_RETINAFACE = True

# Output-Dir anlegen, falls noch nicht vorhanden
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_config():
    """Liest die DeepFace-Settings aus der config.yaml."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            for model in cfg.get("pipeline", []):
                if model.get("id") == "deepface":
                    return model
    except Exception as e:
        print(f"Fehler beim Laden der Config: {e}")

    # Fallback-Werte
    return {
        "enabled": True,
        "use_retinaface": USE_RETINAFACE,
        "Deepface_emotion": True,
        "Deepface_alter": True,
        "Deepface_geschlecht": True,
    }


def _round_or_none(value):
    try:
        return round(float(value), 4)
    except Exception:
        return None


def _confidence_for_label(score_dict, label):
    if not isinstance(score_dict, dict) or not label:
        return None
    return _round_or_none(score_dict.get(label))


def _map_gender_to_de(dominant_gender):
    if dominant_gender == "Man":
        return "Mann"
    if dominant_gender == "Woman":
        return "Frau"
    return dominant_gender


print(f"DeepFace Worker aktiv. Ueberwache: {INPUT_DIR}")
last_backend = None

while True:
    # 1. Aktuelle Konfiguration pruefen
    config = load_config()
    if not config.get("enabled", True):
        time.sleep(5)
        continue
    use_retinaface = bool(config.get("use_retinaface", USE_RETINAFACE))
    detector_backend = "retinaface" if use_retinaface else "skip"
    if detector_backend != last_backend:
        print(f"DeepFace detector_backend aktiv: {detector_backend}")
        last_backend = detector_backend

    # 2. Eingangsordner nach Bildern scannen
    try:
        files = os.listdir(INPUT_DIR)
        valid_files = sorted([
            f for f in files
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ])
    except Exception as e:
        print(f"Fehler beim Scan: {e}")
        time.sleep(2)
        continue

    # 3. Dateien nacheinander verarbeiten
    for filename in valid_files:
        img_path = os.path.join(INPUT_DIR, filename)
        face_id = os.path.splitext(filename)[0]

        yaml_path = os.path.join(PROCESSED_DIR, f"{face_id}_deepface.yaml")

        if os.path.exists(yaml_path):
            os.remove(img_path)
            continue

        try:
            print(f"Analysiere {filename}...")
            started_at = time.perf_counter()

            # Bei deaktiviertem RetinaFace nutzt DeepFace direkt den YOLO-Crop.
            results = DeepFace.analyze(
                img_path=img_path,
                actions=["age", "gender", "emotion"],
                enforce_detection=False,
                detector_backend=detector_backend,
                silent=True,
            )
            res = results[0] if isinstance(results, list) else results

            # Daten zusammenbauen
            out = {}
            dominant_emotion = res.get("dominant_emotion")
            dominant_gender = res.get("dominant_gender")

            emotion_confidence = _confidence_for_label(res.get("emotion"), dominant_emotion)
            gender_confidence = _confidence_for_label(res.get("gender"), dominant_gender)

            if config.get("Deepface_emotion", True):
                out["Emotion"] = dominant_emotion
            if config.get("Deepface_alter", True):
                out["Alter"] = int(res.get("age", 0))
            if config.get("Deepface_geschlecht", True):
                out["Geschlecht"] = _map_gender_to_de(dominant_gender)

            # Bei detector_backend='skip' ist face_confidence meist nicht aussagekraftig.
            out["Emotion_Confidence"] = emotion_confidence
            out["Gender_Confidence"] = gender_confidence

            # Kompatibilitaetsfeld: bevorzugt Emotion-Confidence, sonst Gender-Confidence.
            out["Confidence"] = (
                emotion_confidence if emotion_confidence is not None else gender_confidence
            )
            elapsed_s = time.perf_counter() - started_at

            print(
                f"[DeepFace] {filename} | "
                f"emotion={out.get('Emotion')} ({emotion_confidence}) | "
                f"gender={out.get('Geschlecht')} ({gender_confidence}) | "
                f"age={out.get('Alter')} | "
                f"confidence={out.get('Confidence')} | "
                f"time={elapsed_s:.3f}s"
            )

            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(out, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                f.flush()
                os.fsync(f.fileno())

            print(f"Gespeichert: {os.path.basename(yaml_path)}")

            # Datei loeschen damit der Loop nicht von vorne beginnt
            os.remove(img_path)

        except Exception as e:
            print(f"DeepFace Fehler bei {filename}: {e}")
            if os.path.exists(img_path):
                os.remove(img_path)

    # 4. Kurze Pause vor dem naechsten Scan
    time.sleep(1)
