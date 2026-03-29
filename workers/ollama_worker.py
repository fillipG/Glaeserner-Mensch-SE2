"""
OllamaWorker
------------
Dieses Skript ist der lokale dateibasierte Ollama-Worker der Anwendung.
Er ueberwacht die von Moondream erzeugte Inbox und schreibt daraus die finalen
Text-YAMLs in den gemeinsamen final-Ordner.

Zustaendigkeiten:
1. Einlesen der aktuellen Ollama-Konfiguration aus config.yaml.
2. Ueberwachung von `general_ordner/ollama_ai/ollama_inbox`.
3. Erzeugen der finalen Kriminalgeschichte ueber die lokale Ollama-API.
4. Pass-Through-Fall, wenn Ollama in der Config deaktiviert ist.
5. Sicheres Schreiben der Output-YAMLs und Aufraeumen der Inbox-Dateien.

AUTOREN: Florian Hoeft
"""

import os
import re
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from path_service import get_paths
from constants import PipelineStage

REPO_ROOT = Path(__file__).resolve().parents[1]
PATHS = get_paths()
INPUT_DIR = PATHS["ollama_inbox"]
PROCESSED_DIR = PATHS["final"]
CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_MODEL = "qwen2.5:3b"


# =========================================================
# KONFIGURATION UND PROMPT-AUFBAU
# =========================================================

def normalize_single_paragraph(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def load_config():
    """Liest die zentrale config.yaml des Projekts fuer den Worker ein."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"[OLLAMA] Konnte Config nicht lesen: {exc}")
        return {}


def load_ollama_config():
    """
    Liest nur den fuer den Worker relevanten Ollama-Teil aus der Gesamt-Config.
    Dadurch greifen Toggle und Modellwechsel ohne separaten Worker-Neustart.
    """
    # Der Worker nutzt dieselbe Config-Datei wie die GUI, damit ein Toggle von Ollama
    # direkt in der naechsten Scan-Runde wirksam wird.
    config = load_config()
    for model_cfg in config.get("pipeline", []):
        if model_cfg.get("id") == PipelineStage.OLLAMA:
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
    """
    Baut den Prompt fuer die lokale Ollama-Anfrage.
    Im normalen Live-Betrieb basiert der Text nur auf der Moondream-Beschreibung,
    nicht auf den parallel erzeugten DeepFace-Daten.
    """
    description = moondream_data.get("moondream_description", "")
    return (
        f"Personenbeschreibung: {description}\n\n"
        f"{prompt_template}"
    )


# =========================================================
# DATEIVERARBEITUNG
# =========================================================

def process_file(filename, worker_config):
    """
    Standardpfad fuer eine neue Moondream-Zwischen-YAML.
    Wenn Ollama aktiv ist, wird die Datei gelesen, an Ollama uebergeben und
    danach als finales YAML in den final-Ordner geschrieben.
    """
    file_path = INPUT_DIR / filename

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            moondream_data = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"[OLLAMA] Konnte {filename} nicht lesen: {exc}")
        return False

    prompt_template = worker_config.get("prompt", "").strip()
    if not prompt_template:
        return False

    # Wenn Ollama deaktiviert ist, wird die Moondream-Beschreibung nicht verworfen,
    # sondern als finales YAML in den final-Ordner durchgereicht.
    if not worker_config.get("enabled", True):
        return process_file_passthrough(filename, moondream_data)

    # SCHRITT 1: Prompt aus Moondream-Daten und konfiguriertem Template erzeugen
    prompt = build_ollama_prompt(moondream_data, prompt_template)
    model_name = worker_config.get("model", DEFAULT_MODEL) or DEFAULT_MODEL

    # SCHRITT 2: Lokale Ollama-Anfrage ausfuehren
    try:
        import ollama
        print(f"[OLLAMA] Verarbeite {filename} mit Modell {model_name}...")
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        description = normalize_single_paragraph(response["message"]["content"])
    except Exception as exc:
        print(f"[OLLAMA] Fehler bei {filename}: {exc}")
        return False

    # SCHRITT 3: Finales Output-YAML fuer den final-Ordner aufbauen
    output_data = {
        "prompt": prompt_template,
        "source_prompt": moondream_data.get("moondream_prompt", ""),
        "source_description": moondream_data.get("moondream_description", ""),
        "description": description,
    }

    return write_output_and_cleanup(file_path, PROCESSED_DIR / filename, output_data)


def process_file_passthrough(filename, moondream_data):
    """
    Fallback fuer deaktiviertes Ollama.
    Die von Moondream erzeugte Beschreibung wird dann direkt als finales YAML
    in den final-Ordner weitergereicht.
    """
    base_name = Path(filename).stem
    if base_name.endswith("_ollama"):
        base_name = base_name[:-7]

    description = normalize_single_paragraph(moondream_data.get("moondream_description", ""))
    output_path = PROCESSED_DIR / f"{base_name}_moondream.yaml"
    output_data = {
        "prompt": moondream_data.get("moondream_prompt", ""),
        "description": description,
        "source_model": "moondream",
    }
    return write_output_and_cleanup(INPUT_DIR / filename, output_path, output_data)


def write_output_and_cleanup(file_path, output_path, output_data):
    """
    Schreibt das finale YAML und loescht erst danach die Inbox-Datei.
    So sieht der naechste Pipeline-Schritt nie unvollstaendige oder verlorene
    Zwischenergebnisse.
    """
    # Erst nach erfolgreichem Schreiben wird die Inbox-Datei geloescht, damit der naechste
    # Schritt nie auf ein halbfertiges oder verlorenes Ergebnis zeigt.
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
    except OSError:
        # Der nachgelagerte Cleanup kann die Inbox-Datei bereits entfernt haben.
        # Das ist in diesem Ablauf harmlos und wird bewusst still ignoriert.
        pass
    return True


# =========================================================
# WORKER-HAUPTSCHLEIFE
# =========================================================

while True:
    # Die Config wird in jeder Runde neu gelesen, damit Toggle und Prompt-Aenderungen
    # ohne Neustart des Workers wirksam werden.
    worker_config = load_ollama_config()

    try:
        # SCHRITT 1: Arbeitsordner sicherstellen und neue Inbox-Dateien suchen
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        files = os.listdir(INPUT_DIR)
        # Der Worker reagiert nur auf vollstaendig von Moondream geschriebene Zwischen-YAMLs.
        valid_files = sorted([
            file_name for file_name in files
            if file_name.lower().endswith(".yaml")
            and re.match(r"^(?:batch\d+_)?face\d+_ollama\.yaml$", file_name, re.IGNORECASE)
        ])
    except Exception as exc:
        print(f"[OLLAMA] Fehler beim Scan: {exc}")
        time.sleep(2)
        continue

    # SCHRITT 2: Gefundene Moondream-Zwischen-YAMLs nacheinander verarbeiten
    for filename in valid_files:
        process_file(filename, worker_config)

    # Kurze Pause, damit der Worker den Ordner nicht im Busy-Loop scannt
    time.sleep(0.5)
