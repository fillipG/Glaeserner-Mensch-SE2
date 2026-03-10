import os
import yaml


class DescriptionRepository:
    def __init__(self, base_dir="General ordner/final"):
        self.base_dir = base_dir

    def exists(self):
        return os.path.exists(self.base_dir)

    def read_moondream_description(self, index):
        return self._read_description_file(f"face{index + 1}_moondream.yaml")

    def read_ollama_description(self, index):
        return self._read_description_file(f"face{index + 1}_ollama.yaml")

    def read_deepface_data(self, index):
        file_name = f"face{index + 1}_deepface.yaml"
        file_path = os.path.join(self.base_dir, file_name)
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return None

    def _read_description_file(self, file_name):
        file_path = os.path.join(self.base_dir, file_name)
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            return None
        description = data.get("description")
        if isinstance(description, str) and description.strip():
            return description.strip()
        return None

