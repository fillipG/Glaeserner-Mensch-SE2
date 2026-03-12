import copy
import os
import yaml


class ConfigService:
    def __init__(self, path="config.yaml", default_llm_value="qwen2.5:3b"):
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
        self.ensure_defaults(config)
        return config

    def save(self, config):
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=False)

    def get_default_config(self):
        ollama_prompt = (
            "Write a criminal report about a fictional person.\n"
            "The person has already been described. Write only what crime\n"
            "the person might have committed in a short flowing paragraph.\n"
            "Make sure it is a crime within the Stasi context.\n"
            "The output must be between 30 and 50 words long.\n"
            "Stay within this range and try to make it a little funny.\n"
            "The story does not need to be explained. It is enough to show one crime."
        )
        return {
            "language": "de",
            "photo_delay": 3,
            "reset_countdown_seconds": 3,
            "close_on_no_person_enabled": True,
            "close_on_no_person_seconds": 10,
            "no_person_check_interval_ms": 2000,
            "pipeline_timeout_seconds": 300,
            "fullscreen": True,
            "developer_mode": False,
            "animation_speed": 15,
            "llm_model": self.default_llm_value,
            "face_yolo": {
                "confidence": 0.5,
            },
            "pool": {
                "enabled": True,
                "path": "./pool",
                "max_extra_persons": 3,
                "cooldown_batches": 3,
            },
            "pipeline": [
                {
                    "id": "moondream",
                    "name": "Visual Description (VLM)",
                    "enabled": True,
                    "final_output": False,
                    "watch_dir": "./General ordner/ollama_ai/ollama_inbox",
                    "file_ext": ".yaml",
                    "show_preview": True,
                    "prompt": "Name the clothing and any accessories the person is wearing. Put in 4 Sentences",
                },
                {
                    "id": "ollama",
                    "name": "Kriminalgeschichte (Ollama)",
                    "enabled": True,
                    "final_output": True,
                    "watch_dir": "./General ordner/final",
                    "file_ext": ".yaml",
                    "prompt": ollama_prompt,
                },
                {
                    "id": "deepface",
                    "name": "Emotionserkennung",
                    "enabled": True,
                    "final_output": True,
                    "watch_dir": "./General ordner/final",
                    "file_ext": ".yaml",
                    "use_retinaface": True,
                },
                {
                    "id": "fer",
                    "name": "Emotionserkennung (FER)",
                    "enabled": False,
                    "final_output": True,
                    "watch_dir": "./General ordner/final",
                    "file_ext": ".yaml",
                },
            ],
        }

    def get_default_admin_settings(self):
        defaults = self.get_default_config()
        pipeline_defaults = {entry["id"]: entry for entry in defaults["pipeline"]}
        pool_defaults = defaults["pool"]
        return {
            "photo_delay": defaults["photo_delay"],
            "close_on_no_person_enabled": defaults["close_on_no_person_enabled"],
            "close_on_no_person_seconds": defaults["close_on_no_person_seconds"],
            "animation_speed": defaults["animation_speed"],
            "pipeline_timeout_seconds": defaults["pipeline_timeout_seconds"],
            "face_yolo_confidence": defaults["face_yolo"]["confidence"],
            "fullscreen": defaults["fullscreen"],
            "developer_mode": defaults["developer_mode"],
            "pool_enabled": pool_defaults["enabled"],
            "pool_max_extra_persons": pool_defaults["max_extra_persons"],
            "pool_cooldown_batches": pool_defaults["cooldown_batches"],
            "moondream_enabled": pipeline_defaults["moondream"]["enabled"],
            "moondream_prompt": pipeline_defaults["moondream"]["prompt"],
            "ollama_enabled": pipeline_defaults["ollama"]["enabled"],
            "ollama_prompt": pipeline_defaults["ollama"]["prompt"],
            "deepface_enabled": pipeline_defaults["deepface"]["enabled"],
            "deepface_use_retinaface": pipeline_defaults["deepface"]["use_retinaface"],
            "fer_enabled": pipeline_defaults["fer"]["enabled"],
            "llm_model": defaults["llm_model"],
        }

    def ensure_defaults(self, config):
        # Fehlende Standardwerte ergaenzen, ohne bestehende Laufzeitwerte zu ueberschreiben.
        legacy_wait_time_file_closed = config.pop("wait_time_file_closed", None)
        if legacy_wait_time_file_closed is not None:
            config["photo_delay"] = legacy_wait_time_file_closed

        legacy_face_yolo_confidence = config.pop("face_yolo_confidence", None)
        if legacy_face_yolo_confidence is not None:
            face_yolo = config.get("face_yolo")
            if not isinstance(face_yolo, dict):
                face_yolo = {}
                config["face_yolo"] = face_yolo
            face_yolo.setdefault("confidence", legacy_face_yolo_confidence)

        defaults = self.get_default_config()
        self._merge_dict_defaults(config, defaults, skip_keys={"pipeline"})
        self._merge_pipeline_defaults(config, defaults["pipeline"])

    def ensure_base_defaults(self, config):
        self.ensure_defaults(config)

    def reset_admin_settings(self, config):
        # Nur Admin-Einstellungen gezielt auf die zentral definierten Code-Defaults zuruecksetzen.
        defaults = self.get_default_config()
        pipeline_defaults = {entry["id"]: entry for entry in defaults["pipeline"]}

        config["photo_delay"] = defaults["photo_delay"]
        config["close_on_no_person_enabled"] = defaults["close_on_no_person_enabled"]
        config["close_on_no_person_seconds"] = defaults["close_on_no_person_seconds"]
        config["animation_speed"] = defaults["animation_speed"]
        config["pipeline_timeout_seconds"] = defaults["pipeline_timeout_seconds"]
        config["fullscreen"] = defaults["fullscreen"]
        config["developer_mode"] = defaults["developer_mode"]
        config["llm_model"] = defaults["llm_model"]

        face_yolo = config.get("face_yolo")
        if not isinstance(face_yolo, dict):
            face_yolo = {}
            config["face_yolo"] = face_yolo
        face_yolo["confidence"] = defaults["face_yolo"]["confidence"]
        config.pop("face_yolo_confidence", None)

        pool = config.get("pool")
        if not isinstance(pool, dict):
            pool = {}
            config["pool"] = pool
        pool["enabled"] = defaults["pool"]["enabled"]
        pool["max_extra_persons"] = defaults["pool"]["max_extra_persons"]
        pool["cooldown_batches"] = defaults["pool"]["cooldown_batches"]

        self._merge_pipeline_defaults(config, defaults["pipeline"])
        for model_id, keys in {
            "moondream": ("enabled", "prompt"),
            "ollama": ("enabled", "prompt"),
            "deepface": ("enabled", "use_retinaface"),
            "fer": ("enabled",),
        }.items():
            entry = self.get_pipeline_entry(config, model_id)
            if entry is None:
                continue
            for key in keys:
                entry[key] = copy.deepcopy(pipeline_defaults[model_id][key])

        return config


    def get_pipeline_entry(self, config, model_id):
        pipeline = config.get("pipeline", [])
        if not isinstance(pipeline, list):
            return None
        return next((entry for entry in pipeline if isinstance(entry, dict) and entry.get("id") == model_id), None)

    def _merge_dict_defaults(self, target, defaults, skip_keys=None):
        skip_keys = set(skip_keys or ())
        for key, default_value in defaults.items():
            if key in skip_keys:
                continue
            current_value = target.get(key)
            if isinstance(default_value, dict):
                if not isinstance(current_value, dict):
                    target[key] = copy.deepcopy(default_value)
                else:
                    self._merge_dict_defaults(current_value, default_value)
            elif isinstance(default_value, list):
                if not isinstance(current_value, list):
                    target[key] = copy.deepcopy(default_value)
            else:
                target.setdefault(key, default_value)

    def _merge_pipeline_defaults(self, config, pipeline_defaults):
        pipeline = config.get("pipeline")
        if not isinstance(pipeline, list):
            pipeline = []
            config["pipeline"] = pipeline

        for default_entry in pipeline_defaults:
            entry = self.get_pipeline_entry(config, default_entry["id"])
            if entry is None:
                pipeline.append(copy.deepcopy(default_entry))
                continue
            self._merge_dict_defaults(entry, default_entry)
