import time
import torch
import os
import re
import yaml
from transformers import AutoModelForCausalLM
from PIL import Image

# Pfade innerhalb des Docker-Containers
INPUT_DIR = "/data/input"
PROCESSED_DIR = "/data/processed"
CONFIG_PATH = "/app/config.yaml"


def get_latest_prompt():
    try:
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
            for model_cfg in config.get("pipeline", []):
                if model_cfg["id"] == "moondream":
                    return model_cfg.get("prompt", "Describe the person.")
    except Exception as e:
        print(f"--- Could not read config, using fallback: {e} ---")
    return "Describe the person in detail."


print("--- Loading Model into VRAM... ---")
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda",
)

last_used_prompt = ""
processed_in_session = set()

print(f"--- Worker started. Watching {INPUT_DIR} ---")

while True:
    # 1. Prompt aus Config laden
    current_prompt = get_latest_prompt()

    # Falls der Prompt geändert wurde, Cache leeren (für Re-Processing)
    if current_prompt != last_used_prompt:
        print(f"--- Prompt changed to: {current_prompt} ---")
        last_used_prompt = current_prompt
        processed_in_session = set()

    # 2. Dateien im Ordner scannen
    try:
        all_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        valid_files = [f for f in all_files if re.match(r'^face\d+', f, re.IGNORECASE)]
    except Exception as e:
        print(f"❌ Error scanning directory: {e}")
        time.sleep(2)
        continue

    if not valid_files:
        time.sleep(1)
        continue

    for filename in valid_files:
        # Pfade generieren
        img_path = os.path.join(INPUT_DIR, filename)
        name_part = os.path.splitext(filename)[0]
        yaml_filename = f"{name_part}_moondream.yaml"
        yaml_path = os.path.join(PROCESSED_DIR, yaml_filename)

        # PRÜFUNG: Nur verarbeiten, wenn noch kein Ergebnis existiert
        # (oder es in dieser Sitzung nach Prompt-Wechsel noch nicht dran war)
        if filename in processed_in_session or os.path.exists(yaml_path):
            continue

        try:
            print(f"Processing {filename}...")
            image = Image.open(img_path).convert("RGB")

            # KI Abfrage
            answer = model.query(image, current_prompt)["answer"]

            # Ergebnis-Daten vorbereiten
            output_data = {
                "description": answer.strip()
            }

            # YAML Datei schreiben
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    output_data,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True
                )

            processed_in_session.add(filename)
            print(f"✅ Created {yaml_filename}")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            time.sleep(2)

    # Kurze Pause vor dem nächsten Scan
    time.sleep(1)