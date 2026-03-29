"""
Moondream - Vision Language Model (VLM)
---------------------------------------
Bildanalyse-Worker fuer die Museums-Pipeline.

Der Worker:
1. Ueberwacht die Moondream-Inbox auf neue Personenbilder.
2. Analysiert das Bild mit GPU-Beschleunigung.
3. Uebergibt die Beschreibung als YAML an die Ollama-Inbox.

AUTOR: Dennis Penner
"""

import json
import os
import re
import shutil
import threading
import time
from datetime import datetime, timedelta, timezone

import torch
import yaml
from PIL import Image
from transformers import AutoModelForCausalLM


INPUT_DIR = "/app/moondream_inbox"
OLLAMA_INBOX_DIR = "/app/ollama_inbox"
CONFIG_PATH = "/app/config.yaml"
FAILED_DIR = "/app/failed"
STATUS_DIR = "/app/status"
HEARTBEAT_PATH = os.path.join(STATUS_DIR, "heartbeat.json")
DEFAULT_MODEL_ID = "vikhyatk/moondream2"
DEFAULT_MODEL_REVISION = "6b714b26eea5cbd9f31e4edb2541c170afa935ba"
FAILED_RETENTION_DAYS = 14
FAILED_RETENTION_INTERVAL_SECONDS = 3600
PROCESSING_TIMEOUT_SECONDS = float(os.environ.get("MOONDREAM_PROCESSING_TIMEOUT_SECONDS", "900"))


def load_moondream_config():
    """Liest die zentrale Konfiguration ein."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
            for model_cfg in config.get("pipeline", []):
                if model_cfg.get("id") == "moondream":
                    return {
                        "enabled": bool(model_cfg.get("enabled", True)),
                        "prompt": model_cfg.get("prompt", "Describe the person."),
                    }
    except Exception as exc:
        print(f"--- Konnte Config nicht lesen, nutze Fallback: {exc} ---")
    return {
        "enabled": True,
        "prompt": "Describe the person in detail.",
    }


def load_model():
    """Laedt das gepinnte Moondream-Modell oder beendet den Worker sauber."""
    model_id = os.environ.get("MOONDREAM_MODEL_ID", DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID
    model_revision = (
        os.environ.get("MOONDREAM_REVISION", DEFAULT_MODEL_REVISION).strip()
        or DEFAULT_MODEL_REVISION
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"--- Lade Moondream Modell... ({model_id}@{model_revision}) ---")
    print(f"--- Device: {device} | dtype: {dtype} ---")
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=model_revision,
            trust_remote_code=True,
            dtype=dtype,
            device_map=device,
        )
    except Exception as exc:
        print(f"--- Moondream Modell konnte nicht geladen werden: {exc} ---")
        raise SystemExit(1)


def prune_failed_dir():
    """Entfernt alte Fehlerartefakte, damit failed/ nicht unendlich waechst."""
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
        "source_model": "moondream",
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
    """Beendet den Prozess, wenn processing zu lange haengt."""
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
                f"Moondream Heartbeat-Timeout: processing seit {heartbeat_age:.1f}s "
                f"(Limit {PROCESSING_TIMEOUT_SECONDS:.1f}s)."
            )
            os._exit(1)


write_heartbeat("startup")
threading.Thread(target=heartbeat_watchdog, daemon=True).start()
model = load_model()
last_used_prompt = ""
last_retention_run = 0.0
os.makedirs(FAILED_DIR, exist_ok=True)
prune_failed_dir()
enable_runtime_monitor()
write_heartbeat("ready")
print(f"--- Moondream aktiv. Ueberwache Inbox: {INPUT_DIR} ---")


while True:
    if time.time() - last_retention_run >= FAILED_RETENTION_INTERVAL_SECONDS:
        prune_failed_dir()
        last_retention_run = time.time()

    moondream_config = load_moondream_config()
    if not moondream_config.get("enabled", True):
        write_heartbeat("idle")
        time.sleep(2)
        continue

    current_prompt = moondream_config.get("prompt", "Describe the person in detail.")
    if current_prompt != last_used_prompt:
        print(f"--- Neuer Prompt erkannt: {current_prompt} ---")
        last_used_prompt = current_prompt

    try:
        os.makedirs(INPUT_DIR, exist_ok=True)
        os.makedirs(OLLAMA_INBOX_DIR, exist_ok=True)
        os.makedirs(FAILED_DIR, exist_ok=True)
        all_files = [name for name in os.listdir(INPUT_DIR) if name.lower().endswith((".jpg", ".jpeg", ".png"))]
        valid_files = sorted(
            [name for name in all_files if re.match(r"^(?:batch\d+_)?face\d+", name, re.IGNORECASE)]
        )
        if not valid_files:
            write_heartbeat("idle")
    except Exception as exc:
        write_heartbeat("error")
        print(f"Fehler beim Ordner-Scan: {exc}")
        time.sleep(2)
        continue

    for filename in valid_files:
        img_path = os.path.join(INPUT_DIR, filename)
        name_part = os.path.splitext(filename)[0]
        yaml_filename = f"{name_part}_ollama.yaml"
        yaml_path = os.path.join(OLLAMA_INBOX_DIR, yaml_filename)

        try:
            write_heartbeat("processing")
            print(f"Analysiere {filename}...")
            image = Image.open(img_path).convert("RGB")
            answer = model.query(image, current_prompt)["answer"]
            output_data = {
                "moondream_prompt": current_prompt,
                "moondream_description": answer.strip(),
                "source_model": "moondream",
            }

            with open(yaml_path, "w", encoding="utf-8") as handle:
                yaml.dump(
                    output_data,
                    handle,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
                handle.flush()
                os.fsync(handle.fileno())

            print(f"Analyse fertig: {yaml_filename} -> Weiter an OLLAMA")
            if os.path.exists(img_path):
                os.remove(img_path)
            write_heartbeat("idle")

        except Exception as exc:
            write_heartbeat("error")
            print(f"Fehler bei Analyse von {filename}: {exc}")
            try:
                move_to_failed(img_path, exc)
            except Exception as move_exc:
                print(f"Fehler beim Verschieben nach failed/: {move_exc}")
            write_heartbeat("idle")
            time.sleep(1)

    time.sleep(2)
