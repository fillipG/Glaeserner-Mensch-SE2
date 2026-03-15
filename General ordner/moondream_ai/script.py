"""
Moondream - Vision Language Model (VLM)
---------------------------------------------
Dieses Skript fungiert als Bildanalyse-Einheit. Moondream ist ein spezialisiertes 
KI-Modell, das Bilder "versteht" und auf Basis eines Text-Prompts Beschreibungen 
generiert.

Der Prozess:
1. Überwacht die Moondream-Inbox auf neue Personenfotos.
2. Analysiert das Bild mittels GPU-Beschleunigung (CUDA).
3. Übergibt die Beschreibung als YAML-Datei an die Ollama-Inbox für das Storytelling.

AUTOR: Dennis Penner
"""

import os
import re
import time
import torch
import yaml
from PIL import Image
from transformers import AutoModelForCausalLM

# Definition der Pfade innerhalb der Docker-Container-Struktur.
# Diese Verzeichnisse sind als Volumes gemountet, um den Datenaustausch zwischen
# den isolierten KI-Containern zu ermöglichen.
INPUT_DIR = "/app/moondream_inbox"
OLLAMA_INBOX_DIR = "/app/ollama_inbox"
CONFIG_PATH = "/app/config.yaml"


def load_moondream_config():
    """
    Liest die zentrale Konfiguration ein.
    Ermöglicht es, das Verhalten der KI (z.B. den Fokus der Beschreibung)
    im laufenden Betrieb über den Prompt in der config.yaml zu steuern, ohne den
    Container-Dienst neu starten zu müssen.
    """
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
            # Suche nach dem spezifischen Moondream-Eintrag in der Pipeline-Liste
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


# INITIALISIERUNG.
# Das Modell 'moondream2' wird geladen und direkt in den VRAM der Grafikkarte geschoben.
print("--- Lade Moondream Modell in den VRAM... ---")
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True,       # Erforderlich für die spezifische Modellarchitektur
    dtype=torch.bfloat16,         # Nutzt bfloat16 für effizientere GPU Nutzung
    device_map="cuda",            # Erzwingt die Ausführung auf der NVIDIA-Grafikkarte
)

last_used_prompt = ""
print(f"--- Moondream aktiv. Ueberwache Inbox: {INPUT_DIR} ---")

# HAUPTSCHLEIFE: Der Worker bleibt aktiv und wartet auf Bildmaterial.
while True:
    moondream_config = load_moondream_config()

    # Falls das Modell in der Config deaktiviert wurde, pausiert der Loop.
    if not moondream_config.get("enabled", True):
        time.sleep(2)
        continue

    # Übernimmt den Prompt aus der Config oder nutzt einen Standardwert.
    current_prompt = moondream_config.get("prompt", "Describe the person in detail.")

    # Erkennt Prompt-Änderungen in der config.yaml zur Laufzeit.
    # Ermöglicht Live-Tuning der Bildbeschreibung ohne Neustart des Modells.
    if current_prompt != last_used_prompt:
        print(f"--- Neuer Prompt erkannt: {current_prompt} ---")
        last_used_prompt = current_prompt

    try:
        # Sicherstellen, dass die Verzeichnisstruktur für den Datenaustausch existiert.
        os.makedirs(INPUT_DIR, exist_ok=True)
        os.makedirs(OLLAMA_INBOX_DIR, exist_ok=True)

        # Scannt den Eingangsordner nach Bilddateien, die vom YOLO-Modell
        # dort abgelegt wurden (Namensschema: face_0.jpg, face_1.jpg etc.).
        all_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        valid_files = [f for f in all_files if re.match(r"^body\d+", f, re.IGNORECASE)]
    except Exception as exc:
        print(f"Fehler beim Ordner-Scan: {exc}")
        time.sleep(2)
        continue

    for filename in valid_files:
        img_path = os.path.join(INPUT_DIR, filename)
        name_part = os.path.splitext(filename)[0]  # Extrahiert die ID, z.B. "face_0"

        # Zieldatei für den nächsten Pipeline-Schritt (Ollama-Modell).
        yaml_filename = f"{name_part}_ollama.yaml"
        yaml_path = os.path.join(OLLAMA_INBOX_DIR, yaml_filename)

        try:
            print(f"Analysiere {filename}...")
            # Bild laden und in das RGB-Format konvertieren für die KI-Verarbeitung.
            image = Image.open(img_path).convert("RGB")

            # Das Modell "sieht" das Bild und generiert eine Antwort
            # basierend auf dem in der Config definierten Prompt.
            answer = model.query(image, current_prompt)["answer"]

            # Verpacken der KI-Ergebnisse in ein strukturiertes YAML-Format.
            # Diese Daten dienen als Input für das LLM (Ollama), um daraus
            # die finale Kriminalgeschichte zu generieren.
            output_data = {
                "moondream_prompt": current_prompt,
                "moondream_description": answer.strip(),
                "source_model": "moondream",
            }

            # Schreiben der Datei: Flush und fsync stellen sicher,
            # dass die Datei vollständig auf der Festplatte landet, bevor
            # der nächste Worker (Ollama) sie liest.
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

            print(f"Analyse fertig: {yaml_filename} -> Weiter an OLLAMA")

            # Cleanup: Das verarbeitete Bild wird gelöscht, um Platz zu sparen
            # und eine erneute Verarbeitung zu verhindern.
            if os.path.exists(img_path):
                os.remove(img_path)

        except Exception as exc:
            # Bei Problemen wird das Bild ebenfalls entfernt,
            # um die Pipeline nicht zu blockieren.
            print(f"Fehler bei Analyse von {filename}: {exc}")
            if os.path.exists(img_path):
                os.remove(img_path)
            time.sleep(1)

    # Kurze Pause zur Entlastung der CPU und SSD
    time.sleep(2)