import json
import os
import shutil
import threading
import time
from datetime import datetime, timedelta, timezone

import yaml
from deepface import DeepFace

"""
DeepFace-Worker fuer die Museums-Pipeline.
"""

INPUT_DIR = "/app/deepface_inbox"
PROCESSED_DIR = "/app/final"
CONFIG_PATH = "/app/config.yaml"
FAILED_DIR = "/app/failed"
STATUS_DIR = "/app/status"
HEARTBEAT_PATH = os.path.join(STATUS_DIR, "heartbeat.json")
USE_RETINAFACE = True
FAILED_RETENTION_DAYS = 14
FAILED_RETENTION_INTERVAL_SECONDS = 3600
PROCESSING_TIMEOUT_SECONDS = float(os.environ.get("DEEPFACE_PROCESSING_TIMEOUT_SECONDS", "120"))

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FAILED_DIR, exist_ok=True)


def load_config():
    """Liest die DeepFace-Settings aus der config.yaml."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
            for model in cfg.get("pipeline", []):
                if model.get("id") == "deepface":
                    return model
    except Exception as exc:
        print(f"Fehler beim Laden der Config: {exc}")

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


def prune_failed_dir():
    """Entfernt alte Fehlerartefakte aus failed/."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=FAILED_RETENTION_DAYS)
    if not os.path.isdir(FAILED_DIR):
        return

    for entry in os.scandir(FAILED_DIR):
        try:
            modified = datetime.fromtimestamp(entry.stat().st_mtime, tz=timezone.utc)
            if modified < cutoff and entry.is_file():
                os.remove(entry.path)
        except Exception as exc:
            print(f"Retention-Fehler in failed/: {entry.path} ({exc})")


def move_to_failed(source_path, error_message):
    """Verschiebt eine fehlgeschlagene Eingabedatei atomar nach failed/."""
    os.makedirs(FAILED_DIR, exist_ok=True)
    file_name = os.path.basename(source_path)
    stem, extension = os.path.splitext(file_name)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target_name = file_name
    target_path = os.path.join(FAILED_DIR, target_name)
    if os.path.exists(target_path):
        target_name = f"{stem}_{timestamp}{extension}"
        target_path = os.path.join(FAILED_DIR, target_name)

    if os.path.exists(source_path):
        shutil.move(source_path, target_path)

    error_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_model": "deepface",
        "input_file": file_name,
        "failed_file": os.path.basename(target_path),
        "error": str(error_message),
    }
    error_path = os.path.join(FAILED_DIR, f"{os.path.splitext(target_name)[0]}_error.yaml")
    with open(error_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(error_payload, handle, sort_keys=False, allow_unicode=True)


_heartbeat_lock = threading.Lock()
_heartbeat_state = "startup"
_heartbeat_timestamp = 0.0
_heartbeat_monitor_enabled = False


def write_heartbeat(state):
    """Schreibt den aktuellen Worker-Status atomar nach /app/status."""
    global _heartbeat_state, _heartbeat_timestamp
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "state": state,
        "pid": os.getpid(),
    }
    os.makedirs(STATUS_DIR, exist_ok=True)
    temp_target = HEARTBEAT_PATH + ".tmp"
    with _heartbeat_lock:
        with open(temp_target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(temp_target, HEARTBEAT_PATH)
        _heartbeat_state = state
        _heartbeat_timestamp = time.time()


def enable_runtime_monitor():
    global _heartbeat_monitor_enabled
    with _heartbeat_lock:
        _heartbeat_monitor_enabled = True


def heartbeat_watchdog():
    """Beendet den Prozess, wenn DeepFace bei einer Datei haengen bleibt."""
    while True:
        time.sleep(5)
        with _heartbeat_lock:
            monitor_enabled = _heartbeat_monitor_enabled
            state = _heartbeat_state
            heartbeat_age = time.time() - _heartbeat_timestamp if _heartbeat_timestamp else 0.0

        if monitor_enabled and state == "processing" and heartbeat_age > PROCESSING_TIMEOUT_SECONDS:
            try:
                write_heartbeat("error")
            except Exception:
                pass
            print(
                f"DeepFace Heartbeat-Timeout: processing seit {heartbeat_age:.1f}s "
                f"(Limit {PROCESSING_TIMEOUT_SECONDS:.1f}s)."
            )
            os._exit(1)


write_heartbeat("startup")
threading.Thread(target=heartbeat_watchdog, daemon=True).start()
print(f"DeepFace Worker aktiv. Ueberwache: {INPUT_DIR}")
last_backend = None
last_retention_run = 0.0
prune_failed_dir()
enable_runtime_monitor()
write_heartbeat("ready")


while True:
    if time.time() - last_retention_run >= FAILED_RETENTION_INTERVAL_SECONDS:
        prune_failed_dir()
        last_retention_run = time.time()

    config = load_config()
    if not config.get("enabled", True):
        write_heartbeat("idle")
        time.sleep(5)
        continue

    use_retinaface = bool(config.get("use_retinaface", USE_RETINAFACE))
    detector_backend = "retinaface" if use_retinaface else "skip"
    if detector_backend != last_backend:
        print(f"DeepFace detector_backend aktiv: {detector_backend}")
        last_backend = detector_backend

    try:
        files = os.listdir(INPUT_DIR)
        os.makedirs(FAILED_DIR, exist_ok=True)
        valid_files = sorted(
            [name for name in files if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        )
        if not valid_files:
            write_heartbeat("idle")
    except Exception as exc:
        write_heartbeat("error")
        print(f"Fehler beim Scan: {exc}")
        time.sleep(2)
        continue

    for filename in valid_files:
        img_path = os.path.join(INPUT_DIR, filename)
        face_id = os.path.splitext(filename)[0]
        yaml_path = os.path.join(PROCESSED_DIR, f"{face_id}_deepface.yaml")

        if os.path.exists(yaml_path):
            os.remove(img_path)
            write_heartbeat("idle")
            continue

        try:
            write_heartbeat("processing")
            print(f"Analysiere {filename}...")
            started_at = time.perf_counter()

            results = DeepFace.analyze(
                img_path=img_path,
                actions=["age", "gender", "emotion"],
                enforce_detection=False,
                detector_backend=detector_backend,
                silent=True,
            )
            res = results[0] if isinstance(results, list) else results

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

            out["Emotion_Confidence"] = emotion_confidence
            out["Gender_Confidence"] = gender_confidence
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

            with open(yaml_path, "w", encoding="utf-8") as handle:
                yaml.dump(out, handle, default_flow_style=False, sort_keys=False, allow_unicode=True)
                handle.flush()
                os.fsync(handle.fileno())

            print(f"Gespeichert: {os.path.basename(yaml_path)}")
            os.remove(img_path)
            write_heartbeat("idle")

        except Exception as exc:
            write_heartbeat("error")
            print(f"DeepFace Fehler bei {filename}: {exc}")
            try:
                move_to_failed(img_path, exc)
            except Exception as move_exc:
                print(f"Fehler beim Verschieben nach failed/: {move_exc}")
            write_heartbeat("idle")

    time.sleep(1)
