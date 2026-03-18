from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent
_PATHS: dict | None = None


def get_paths() -> dict:
    global _PATHS
    if _PATHS is None:
        _PATHS = _load_paths()
    return _PATHS


def reload_paths() -> dict:
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

    return {
        "final": REPO_ROOT / configured_paths.get("final_dir", "General ordner/final"),
        "ollama_inbox": REPO_ROOT / configured_paths.get(
            "ollama_inbox",
            "General ordner/ollama_ai/ollama_inbox",
        ),
        "moondream_inbox": REPO_ROOT / configured_paths.get(
            "moondream_inbox",
            "General ordner/moondream_ai/moondream_inbox",
        ),
        "deepface_inbox": REPO_ROOT / configured_paths.get(
            "deepface_inbox",
            "General ordner/docker-compose-deepface/deepface_inbox",
        ),
        "main_image": REPO_ROOT / configured_paths.get("main_image", "General ordner/main_image"),
        "sketch_dir": REPO_ROOT / configured_paths.get("sketch_dir", "General ordner/sketch"),
    }
