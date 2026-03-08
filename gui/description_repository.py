import os
import yaml


class DescriptionRepository:
    def __init__(self, base_dir="General ordner/final"):
        self.base_dir = base_dir

    def exists(self):
        return os.path.exists(self.base_dir)

    def read_moondream_description(self, index):
        file_name = f"face{index + 1}_moondream.yaml"
        file_path = os.path.join(self.base_dir, file_name)
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            description_lines = []
            found_description = False
            for line in lines:
                clean_line = line.strip()
                if clean_line.startswith("description:"):
                    found_description = True
                    first_part = line.split("description:", 1)[1].strip()
                    if first_part:
                        description_lines.append(first_part)
                    continue
                if found_description:
                    if ":" in clean_line and not line.startswith(" "):
                        break
                    description_lines.append(clean_line)
            if description_lines:
                return " ".join(description_lines).strip()
        except Exception:
            return None
        return None

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

