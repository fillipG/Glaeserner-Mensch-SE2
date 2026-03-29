"""
path_service.py
---------------
Zentrale Pfadauflösung für alle KI-Kommunikationsordner.

Alle Pfade leiten sich aus einem gemeinsamen Basisverzeichnis ab
(Standard: "general_ordner"). Um das gesamte Verzeichnis umzubenennen,
genügt eine einzige Änderung in config.yaml:

    paths:
      base_dir: general_ordner   # oder ein beliebiger anderer Name

Einzelne Pfade können zusätzlich überschrieben werden (relative Pfade
werden dabei relativ zum Projektverzeichnis aufgelöst):

    paths:
      ollama_inbox: custom/path/to/ollama_inbox

Verwendung:
    from path_service import get_paths
    paths = get_paths()
    inbox = paths["ollama_inbox"]  # gibt ein pathlib.Path zurück

WICHTIG: Alle Pfade werden beim ersten Aufruf gecacht. Wenn config.yaml
zur Laufzeit geändert wird, muss reload_paths() aufgerufen werden.
"""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent
_PATHS: dict | None = None


def get_paths() -> dict:
    """
    Gibt das gecachte Pfad-Dictionary zurück.
    Beim ersten Aufruf wird config.yaml gelesen und gecacht.
    """
    global _PATHS
    if _PATHS is None:
        _PATHS = _load_paths()
    return _PATHS


def reload_paths() -> dict:
    """
    Liest config.yaml neu ein und aktualisiert den Cache.
    Wird benötigt wenn der Admin die Pfade zur Laufzeit ändert.
    """
    global _PATHS
    _PATHS = _load_paths()
    return _PATHS


def _load_paths() -> dict:
    config_path = REPO_ROOT / "config.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except Exception:
        config = {}

    configured_paths = config.get("paths", {})
    if not isinstance(configured_paths, dict):
        configured_paths = {}

    # Einzelner Schlüssel für das gesamte Basisverzeichnis.
    # Ändert man hier "general_ordner" auf "general_ordner" (oder beliebig),
    # folgen automatisch ALLE abgeleiteten Pfade mit.
    base_dir = configured_paths.get("base_dir", "general_ordner")

    def resolve(key: str, default_subpath: str) -> Path:
        """
        Gibt den konfigurierten Pfad zurück wenn vorhanden,
        sonst den Standardpfad relativ zum Basisverzeichnis.
        """
        if key in configured_paths:
            return REPO_ROOT / configured_paths[key]
        return REPO_ROOT / base_dir / default_subpath

    return {
        # Pipeline-Ergebnisse (finaler YAML-Output aller KI-Stufen)
        "final": resolve("final_dir", "final"),

        # Eingehende Jobs für den lokalen Ollama-Worker
        "ollama_inbox": resolve("ollama_inbox", "ollama_ai/ollama_inbox"),

        # Eingehende Jobs für den Moondream-Docker-Container
        "moondream_inbox": resolve("moondream_inbox", "moondream_ai/moondream_inbox"),

        # Eingehende Jobs für den DeepFace-Docker-Container
        "deepface_inbox": resolve("deepface_inbox", "docker-compose-deepface/deepface_inbox"),

        # Kamerabild (wird von YOLO geschrieben, von Moondream gelesen)
        "main_image": resolve("main_image", "main_image"),

        # Sketch-Ausgaben (Strichzeichnungs-Vorschau)
        "sketch_dir": resolve("sketch_dir", "sketch"),

        # Gewichte des Face-YOLO-Modells (für Pool-Builder und Live-Preview)
        "face_yolo_weights": resolve(
            "face_yolo_weights",
            "docker-compose-face-Yolo/yolov8n-face.pt",
        ),
    }
