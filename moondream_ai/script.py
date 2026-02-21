import time
import torch
import os
import re
import yaml
from transformers import AutoModelForCausalLM
from PIL import Image

# Pfade innerhalb des Docker-Containers
INPUT_DIR = "/data/input"          # Hier landen die Bilder von YOLO
PROCESSED_DIR = "/data/processed"  # Hier schreibt Moondream die Ergebnisse
CONFIG_PATH = "/app/config.yaml"   # Konfiguration für Prompts

def get_latest_prompt():
    """
    Liest den aktuellen Prompt für Moondream aus der config.yaml aus.
    Ermöglicht Änderungen zur Laufzeit, ohne den Container neu zu starten.
    """
    try:
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
            for model_cfg in config.get("pipeline", []):
                if model_cfg["id"] == "moondream":
                    return model_cfg.get("prompt", "Describe the person.")
    except Exception as e:
        print(f"--- Konnte Config nicht lesen, nutze Fallback: {e} ---")
    return "Describe the person in detail."

# Modell laden
print("--- Lade Moondream Modell in den VRAM (GPU)... ---")
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True,
    dtype=torch.bfloat16, # Optimierung für GPU-Speicher
    device_map="cuda",    # Erzwingt die Nutzung der NVIDIA-Grafikkarte
)

last_used_prompt = ""

print(f"--- Moondream Worker aktiv. Überwache: {INPUT_DIR} ---")

# Hauptschleife
while True:
    # 1. Aktuellen Prompt prüfen (falls er in der Config geändert wurde)
    current_prompt = get_latest_prompt()
    if current_prompt != last_used_prompt:
        print(f"--- Neuer Prompt erkannt: {current_prompt} ---")
        last_used_prompt = current_prompt

    # 2. Eingangsordner nach Bildern scannen
    try:
        all_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # Filter: Wir verarbeiten nur Dateien, die mit "face" beginnen (z.B. face1.jpg)
        valid_files = [f for f in all_files if re.match(r'^face\d+', f, re.IGNORECASE)]
    except Exception as e:
        print(f"Fehler beim Ordner-Scan: {e}")
        time.sleep(2)
        continue

    # 3. Dateien nacheinander verarbeiten
    for filename in valid_files:
        img_path = os.path.join(INPUT_DIR, filename)
        name_part = os.path.splitext(filename)[0]
        yaml_filename = f"{name_part}_moondream.yaml"
        yaml_path = os.path.join(PROCESSED_DIR, yaml_filename)

        # Überspringen, wenn das Ergebnis bereits existiert.
        if os.path.exists(yaml_path):
            continue

        try:
            print(f"Analysiere {filename}...")
            image = Image.open(img_path).convert("RGB")

            # KI-Abfrage (Inference)
            answer = model.query(image, current_prompt)["answer"]

            # Ergebnis-Daten strukturieren
            output_data = {
                "prompt": current_prompt,
                "description": answer.strip()
            }

            # Als YAML-Datei speichern
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    output_data,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True
                )
                # 'flush' und 'fsync' erzwingen das Schreiben auf die Festplatte
                # damit der PipelineManager auf Windows die Datei sofort sieht.
                f.flush()
                os.fsync(f.fileno())

            print(f"✅ Analyse fertig: {yaml_filename}")

        except Exception as e:
            print(f"Fehler bei Analyse von {filename}: {e}")
            time.sleep(2)

    # 4. Kurze Pause vor dem nächsten Scan
    time.sleep(1)