import os
import re
import time

import torch
import yaml
from PIL import Image
from transformers import AutoModelForCausalLM

INPUT_DIR = "/app/moondream_inbox"
OLLAMA_INBOX_DIR = "/app/ollama_inbox"
CONFIG_PATH = "/app/config.yaml"


def load_moondream_config():
    # Der Worker liest die Config zyklisch neu ein, damit Prompt- und Enable-Aenderungen
    # ohne Container-Neustart wirksam werden.
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
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


print("--- Lade Moondream Modell in den VRAM (GPU)... ---")
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda",
)

last_used_prompt = ""
print(f"--- Moondream Worker aktiv. Ueberwache Inbox: {INPUT_DIR} ---")

while True:
    moondream_config = load_moondream_config()
    if not moondream_config.get("enabled", True):
        time.sleep(2)
        continue

    current_prompt = moondream_config.get("prompt", "Describe the person in detail.")
    if current_prompt != last_used_prompt:
        print(f"--- Neuer Prompt erkannt: {current_prompt} ---")
        last_used_prompt = current_prompt

    try:
        os.makedirs(INPUT_DIR, exist_ok=True)
        os.makedirs(OLLAMA_INBOX_DIR, exist_ok=True)
        all_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        valid_files = [f for f in all_files if re.match(r"^face\d+", f, re.IGNORECASE)]
    except Exception as exc:
        print(f"Fehler beim Ordner-Scan: {exc}")
        time.sleep(2)
        continue

    for filename in valid_files:
        img_path = os.path.join(INPUT_DIR, filename)
        name_part = os.path.splitext(filename)[0]
        print(f"[{name_part.upper()}] Picked up by MOONDREAM")
        yaml_filename = f"{name_part}_ollama.yaml"
        yaml_path = os.path.join(OLLAMA_INBOX_DIR, yaml_filename)

        try:
            print(f"Analysiere {filename}...")
            image = Image.open(img_path).convert("RGB")
            answer = model.query(image, current_prompt)["answer"]
            print(f"[{name_part.upper()}] MOONDREAM finished")
            # Moondream liefert immer nur die Zwischenbeschreibung fuer den naechsten Schritt.
            # Die Entscheidung, ob Ollama daraus einen Endtext macht oder nur durchreicht,
            # passiert spaeter im lokalen Ollama-Worker.
            output_data = {
                "moondream_prompt": current_prompt,
                "moondream_description": answer.strip(),
                "source_model": "moondream",
            }

            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    output_data,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
                f.flush()
                os.fsync(f.fileno())

            print(f"Analyse fertig: {yaml_filename}")
            print(f"[{name_part.upper()}] Sent to OLLAMA")
            if os.path.exists(img_path):
                os.remove(img_path)
                print(f"Inbox geleert: {filename}")

        except Exception as exc:
            print(f"Fehler bei Analyse von {filename}: {exc}")
            if os.path.exists(img_path):
                os.remove(img_path)
            time.sleep(1)

    time.sleep(0.5)