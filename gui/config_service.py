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
        config.setdefault("close_on_no_person_enabled", True)
        config.setdefault("close_on_no_person_seconds", 10)
        config.setdefault("no_person_check_interval_ms", 2000)
        config.setdefault("pipeline_timeout_seconds", 30)
        config.setdefault("fullscreen", True)
        config.setdefault("developer_mode", False)
        config.setdefault("llm_model", self.default_llm_value)
        legacy_face_yolo_confidence = config.pop("face_yolo_confidence", None)
        face_yolo = config.setdefault("face_yolo", {})
        if not isinstance(face_yolo, dict):
            face_yolo = {}
            config["face_yolo"] = face_yolo
        if legacy_face_yolo_confidence is not None:
            face_yolo.setdefault("confidence", legacy_face_yolo_confidence)
        face_yolo.setdefault("confidence", 0.5)

        pool = config.setdefault("pool", {})
        if not isinstance(pool, dict):
            pool = {}
            config["pool"] = pool
        pool.setdefault("enabled", True)
        pool.setdefault("path", "./pool")
        pool.setdefault("max_extra_persons", 3)
        pool.setdefault("cooldown_batches", 3)

