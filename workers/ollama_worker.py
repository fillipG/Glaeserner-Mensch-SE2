import os
import re
import time
from pathlib import Path

import ollama
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_ROOT / "General ordner" / "ollama_ai" / "ollama_inbox"
PROCESSED_DIR = REPO_ROOT / "General ordner" / "final"
CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_MODEL = "qwen2.5:3b"


def load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"[OLLAMA] Konnte Config nicht lesen: {exc}")
        return {}


def load_ollama_config():
    config = load_config()
    for model_cfg in config.get("pipeline", []):
        if model_cfg.get("id") == "ollama":
            return {
                "enabled": bool(model_cfg.get("enabled", True)),
                "prompt": model_cfg.get("prompt", ""),
                "model": config.get("llm_model", DEFAULT_MODEL),
            }
    return {
        "enabled": True,
        "prompt": "",
        "model": config.get("llm_model", DEFAULT_MODEL),
    }


def build_ollama_prompt(moondream_data, prompt_template):
    description = moondream_data.get("moondream_description", "")
    return (
        f"Personenbeschreibung: {description}\n\n"
        f"{prompt_template}"
    )


def process_file(filename, worker_config):
    file_path = INPUT_DIR / filename
    output_path = PROCESSED_DIR / filename

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            moondream_data = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"[OLLAMA] Konnte {filename} nicht lesen: {exc}")
        return False

    prompt_template = worker_config.get("prompt", "").strip()
    if not prompt_template:
        print(f"[OLLAMA] Kein Prompt konfiguriert, ueberspringe {filename}")
        return False

    prompt = build_ollama_prompt(moondream_data, prompt_template)
    model_name = worker_config.get("model", DEFAULT_MODEL) or DEFAULT_MODEL

    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        description = response["message"]["content"].strip()
    except Exception as exc:
        print(f"[OLLAMA] Fehler bei {filename}: {exc}")
        return False

    output_data = {
        "prompt": prompt_template,
        "source_prompt": moondream_data.get("moondream_prompt", ""),
        "source_description": moondream_data.get("moondream_description", ""),
        "description": description,
    }

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(output_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            f.flush()
            os.fsync(f.fileno())
    except Exception as exc:
        print(f"[OLLAMA] Konnte {output_path} nicht schreiben: {exc}")
        return False

    try:
        os.remove(file_path)
    except OSError as exc:
        print(f"[OLLAMA] Konnte Inbox-Datei {filename} nicht loeschen: {exc}")
    return True


print(f"[OLLAMA] Worker aktiv. Ueberwache: {INPUT_DIR}")

while True:
    worker_config = load_ollama_config()
    if not worker_config.get("enabled", True):
        time.sleep(2)
        continue

    try:
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        files = os.listdir(INPUT_DIR)
        valid_files = [
            file_name for file_name in files
            if file_name.lower().endswith(".yaml") and re.match(r"^face\d+_ollama\.yaml$", file_name, re.IGNORECASE)
        ]
    except Exception as exc:
        print(f"[OLLAMA] Fehler beim Scan: {exc}")
        time.sleep(2)
        continue

    for filename in valid_files:
        process_file(filename, worker_config)

    time.sleep(0.5)
