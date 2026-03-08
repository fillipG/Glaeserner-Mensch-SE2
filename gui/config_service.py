import os
import yaml


class ConfigService:
    def __init__(self, path="config.yaml", default_llm_value="llama3.2:1b"):
        self.path = path
        self.default_llm_value = default_llm_value

    def load(self):
        config = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
            except Exception:
                config = {}
        if not isinstance(config, dict):
            config = {}
        self.ensure_base_defaults(config)
        return config

    def save(self, config):
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=False)

    def ensure_base_defaults(self, config):
        config.setdefault("language", "de")
        config.setdefault("wait_time_file_closed", 3)
        config.setdefault("reset_countdown_seconds", 3)
        config.setdefault("fullscreen", True)
        config.setdefault("developer_mode", False)
        config.setdefault("llm_model", self.default_llm_value)

