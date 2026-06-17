"""
config_service.py
-----------------
Zentraler Konfigurations-Service der Anwendung.

Diese Klasse lädt, speichert und normalisiert die config.yaml.
Sie wird von mehreren Modulen genutzt (GUI, PipelineManager, LocalWorkerManager),
deshalb liegt sie im Root-Verzeichnis und nicht in gui/.

Zuständigkeiten:
- config.yaml laden und mit Defaults ergänzen (ensure_defaults)
- config.yaml speichern
- Legacy-Keys migrieren (z.B. wait_time_file_closed → photo_delay)
- Pipeline-Einträge per ID abrufen (get_pipeline_entry)
- Admin-Einstellungen auf Defaults zurücksetzen (reset_admin_settings)

AUTOREN: Fillip Giffhorn, Florian Hoeft
"""

import copy
import os
import tempfile
import yaml
from constants import PipelineStage


class ConfigService:
    """
    Verwalter für die config.yaml.

    Alle Komponenten der Anwendung, die Einstellungen persistieren müssen,
    verwenden diese Klasse. Die Standardwerte sind hier zentral definiert,
    damit ein Zurücksetzen auf Werkseinstellungen immer möglich ist.
    """

    def __init__(self, path="config.yaml", default_llm_value="qwen2.5:3b"):
        """
        :param path: Pfad zur Konfigurationsdatei (Standard: config.yaml im Projektroot).
        :param default_llm_value: Standard-LLM-Modell, das im Admin-Menü vorausgewählt wird.
        """
        self.path = path
        self.default_llm_value = default_llm_value

    def load(self):
        """
        Lädt die Konfiguration und ergänzt fehlende Defaults.
        Wenn die Datei nicht existiert oder beschädigt ist, wird ein leeres
        Default-Dict zurückgegeben.
        :return: Vollständiges Konfigurations-Dictionary.
        """
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
        """
        Speichert die Konfiguration als YAML-Datei.
        :param config: Zu speicherndes Konfigurations-Dictionary.
        """
        target_path = os.path.abspath(self.path)
        target_dir = os.path.dirname(target_path) or "."
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target_dir,
                delete=False,
                suffix=".tmp",
            ) as handle:
                yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
                handle.flush()
                os.fsync(handle.fileno())
                tmp_path = handle.name
            os.replace(tmp_path, target_path)
        except Exception:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

    def get_default_config(self):
        """
        Liefert die zentralen Standardwerte der gesamten Anwendung.
        Dies ist die einzige Stelle, an der Standardwerte definiert werden.
        Änderungen hier wirken sich auf den Reset-Button im Admin-Menü aus.
        :return: Default-Konfiguration als Dictionary.
        """
        # Dieser Prompt wird an Ollama geschickt, wenn keine andere Konfiguration vorliegt.
        # Er definiert den Stil und Umfang der generierten Kriminalgeschichten.
        ollama_prompt = (
            "Write a criminal report about a fictional person.\n"
            "The person has already been described including their appearance, clothing and\n"
            "body language. Write only what crime\n"
            "the person might have committed in a short flowing paragraph.\n"
            "Make sure it is a crime within the Stasi context.\n"
            "The output must be between 30 and 50 words long.\n"
            "Stay within this range and try to make it a little funny.\n"
            "The story does not need to be explained. It is enough to show one crime.\n"
            "Output only a single paragraph with no line breaks."
        )
        return {
            "language": "de",
            "photo_delay": 3,                        # Sekunden Countdown vor Fotoaufnahme
            "reset_countdown_seconds": 3,            # Sekunden für den Reset-Countdown
            "close_on_no_person_enabled": True,      # Auto-Close wenn Person weggeht
            "close_on_no_person_seconds": 10,        # Zeit bis Auto-Close ausgelöst wird
            "no_person_check_interval_ms": 2000,     # Wie oft auf Anwesenheit geprüft wird
            "pipeline_timeout_seconds": 300,         # Max. Wartezeit auf KI-Ergebnisse
            "fullscreen": True,
            "developer_mode": False,
            "animation_speed": 15,                   # ms pro Frame der Ordner-Videos
            "camera": {
                "width": 1280,
                "height": 720,
                "keep_warm": True,
            },
            "sounds": {
                "enabled": True,
                "volume": 0.3,
                "typewriter": "assets/sounds/typewriter_key.wav",
                "folder_open": "assets/sounds/folder_open.wav",
                "folder_close": "assets/sounds/folder_close.wav",
            },
            "live_deepface": {
                "enabled": False,       # Live-Analyse im Wartemodus (Kamera-Preview)
                "interval_seconds": 3,
                "max_faces": 4,
            },
            "llm_model": self.default_llm_value,
            "face_yolo": {
                "confidence": 0.5,  # Mindestsicherheit für Gesichtserkennung (0.1-0.9)
                "max_faces": 4,     # Mehr als 4 Gesichter werden nach Qualitaet begrenzt
                # Moondream-Bildquelle:
                # face = Gesichts-Crop, body = Koerper-Box, body_seg = freigestellter Koerper.
                # shadow bleibt ein versteckter Entwicklermodus, ist aber kein Standard mehr,
                # damit der Modus im Admin-Menue jederzeit sichtbar und umschaltbar bleibt.
                "moondream_crop_mode": "face",
                "body_confidence": 0.35,
                "body_padding_ratio": 0.12,
                "body_fallback_to_face": True,
                "body_matching_required": False,
                "debug_matching": True,
            },
            "pool": {
                "enabled": True,
                "path": "./pool",
                "max_extra_persons": 3,   # Max. Auffüll-Personen aus dem Pool
                "cooldown_batches": 3,    # Wie viele Batches eine Pool-Person pausiert
            },
            "statistics": {
                "enabled": True,
                "retention_days": 365,      # Einträge älter als N Tage werden automatisch gelöscht
            },
            "pipeline": [
                {
                    "id": PipelineStage.MOONDREAM,
                    "name": "Visual Description (VLM)",
                    "enabled": True,
                    "final_output": False,
                    "watch_dir": "./general_ordner/ollama_ai/ollama_inbox",
                    "file_ext": ".yaml",
                    "show_preview": True,
                    # Zusatzteil original in Englisch: Include their clothing, accessories,
                    # body posture, and any suspicious or notable moveements.
                    "prompt": (
                        "Describe this person in 4 sentences as if writing a surveillance report. "
                        "Include their clothing, accessories, body posture, and any suspicious or notable "
                        "moveements."
                    ),
                },
                {
                    "id": PipelineStage.OLLAMA,
                    "name": "Kriminalgeschichte (Ollama)",
                    "enabled": True,
                    "final_output": True,
                    "watch_dir": "./general_ordner/final",
                    "file_ext": ".yaml",
                    # Zusatzteil original in Englisch: Make sure it is a crime within the
                    # Stasi context. The output must be between 30 and 50 words long.
                    # Stay within this range and try to make it a little funny.
                    "prompt": ollama_prompt,
                },
                {
                    "id": PipelineStage.DEEPFACE,
                    "name": "Emotionserkennung",
                    "enabled": True,
                    "final_output": True,
                    "watch_dir": "./general_ordner/final",
                    "file_ext": ".yaml",
                    "use_retinaface": True,
                },
                {
                    "id": PipelineStage.FER,
                    "name": "Emotionserkennung (FER)",
                    "enabled": False,
                    "final_output": True,
                    "watch_dir": "./general_ordner/final",
                    "file_ext": ".yaml",
                },
            ],
        }

    def get_default_admin_settings(self):
        """
        Leitet die Admin-UI-Defaults aus der Standardkonfiguration ab.
        Wird beim Öffnen des Admin-Menüs und beim Reset verwendet.
        :return: Dictionary mit Admin-Einstellungsdefaults.
        """
        defaults = self.get_default_config()
        pipeline_defaults = {entry["id"]: entry for entry in defaults["pipeline"]}
        pool_defaults = defaults["pool"]
        face_yolo_defaults = defaults["face_yolo"]
        return {
            "photo_delay": defaults["photo_delay"],
            "sounds_enabled": defaults["sounds"]["enabled"],
            "sounds_volume": defaults["sounds"]["volume"],
            "close_on_no_person_enabled": defaults["close_on_no_person_enabled"],
            "close_on_no_person_seconds": defaults["close_on_no_person_seconds"],
            "animation_speed": defaults["animation_speed"],
            "pipeline_timeout_seconds": defaults["pipeline_timeout_seconds"],
            "face_yolo_confidence": defaults["face_yolo"]["confidence"],
            "body_yolo_confidence": face_yolo_defaults["body_confidence"],
            "body_padding_ratio": face_yolo_defaults["body_padding_ratio"],
            "moondream_crop_mode": face_yolo_defaults.get("moondream_crop_mode", "face"),
            "fullscreen": defaults["fullscreen"],
            "developer_mode": defaults["developer_mode"],
            "pool_enabled": pool_defaults["enabled"],
            "pool_max_extra_persons": pool_defaults["max_extra_persons"],
            "pool_cooldown_batches": pool_defaults["cooldown_batches"],
            "moondream_enabled": pipeline_defaults[PipelineStage.MOONDREAM]["enabled"],
            "moondream_prompt": pipeline_defaults[PipelineStage.MOONDREAM]["prompt"],
            "ollama_enabled": pipeline_defaults[PipelineStage.OLLAMA]["enabled"],
            "ollama_prompt": pipeline_defaults[PipelineStage.OLLAMA]["prompt"],
            "deepface_enabled": pipeline_defaults[PipelineStage.DEEPFACE]["enabled"],
            "deepface_use_retinaface": pipeline_defaults[PipelineStage.DEEPFACE]["use_retinaface"],
            "fer_enabled": pipeline_defaults[PipelineStage.FER]["enabled"],
            "live_deepface_enabled": defaults["live_deepface"]["enabled"],
            "live_deepface_interval_seconds": defaults["live_deepface"]["interval_seconds"],
            "llm_model": defaults["llm_model"],
            "statistics_enabled": defaults["statistics"]["enabled"],
            "statistics_retention_days": defaults["statistics"]["retention_days"],
        }

    def ensure_defaults(self, config):
        """
        Ergänzt fehlende Werte und migriert veraltete Config-Keys.
        Bestehende Laufzeitwerte werden dabei nicht überschrieben.
        :param config: Zu normalisierendes Konfigurations-Dictionary (wird in-place verändert).
        """
        # Legacy-Migration: Veraltete Keys auf neue Struktur umziehen
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
        """Alias für ensure_defaults (Rückwärtskompatibilität)."""
        self.ensure_defaults(config)

    def reset_admin_settings(self, config):
        """
        Setzt nur die Admin-relevanten Werte auf die Standardkonfiguration zurück.
        Pfade und andere systemkritische Einstellungen bleiben unverändert.
        :param config: Aktuelles Konfigurations-Dictionary (wird in-place verändert).
        :return: Aktualisierte Konfiguration.
        """
        defaults = self.get_default_config()
        pipeline_defaults = {entry["id"]: entry for entry in defaults["pipeline"]}

        config["photo_delay"] = defaults["photo_delay"]
        sounds = config.setdefault("sounds", {})
        if not isinstance(sounds, dict):
            sounds = {}
            config["sounds"] = sounds
        sounds["enabled"] = defaults["sounds"]["enabled"]
        sounds["volume"] = defaults["sounds"]["volume"]
        config["close_on_no_person_enabled"] = defaults["close_on_no_person_enabled"]
        config["close_on_no_person_seconds"] = defaults["close_on_no_person_seconds"]
        config["animation_speed"] = defaults["animation_speed"]
        config["pipeline_timeout_seconds"] = defaults["pipeline_timeout_seconds"]
        config["fullscreen"] = defaults["fullscreen"]
        config["developer_mode"] = defaults["developer_mode"]
        config["llm_model"] = defaults["llm_model"]

        live_deepface = config.get("live_deepface")
        if not isinstance(live_deepface, dict):
            live_deepface = {}
            config["live_deepface"] = live_deepface
        live_deepface["enabled"] = defaults["live_deepface"]["enabled"]
        live_deepface["interval_seconds"] = defaults["live_deepface"]["interval_seconds"]
        live_deepface["max_faces"] = defaults["live_deepface"]["max_faces"]

        face_yolo = config.get("face_yolo")
        if not isinstance(face_yolo, dict):
            face_yolo = {}
            config["face_yolo"] = face_yolo
        face_yolo["confidence"] = defaults["face_yolo"]["confidence"]
        face_yolo["max_faces"] = defaults["face_yolo"]["max_faces"]
        face_yolo["moondream_crop_mode"] = defaults["face_yolo"]["moondream_crop_mode"]
        face_yolo["body_confidence"] = defaults["face_yolo"]["body_confidence"]
        face_yolo["body_padding_ratio"] = defaults["face_yolo"]["body_padding_ratio"]
        face_yolo["body_fallback_to_face"] = defaults["face_yolo"]["body_fallback_to_face"]
        face_yolo["body_matching_required"] = defaults["face_yolo"]["body_matching_required"]
        face_yolo["debug_matching"] = defaults["face_yolo"]["debug_matching"]
        config.pop("face_yolo_confidence", None)

        pool = config.get("pool")
        if not isinstance(pool, dict):
            pool = {}
            config["pool"] = pool
        pool["enabled"] = defaults["pool"]["enabled"]
        pool["max_extra_persons"] = defaults["pool"]["max_extra_persons"]
        pool["cooldown_batches"] = defaults["pool"]["cooldown_batches"]

        statistics = config.get("statistics")
        if not isinstance(statistics, dict):
            statistics = {}
            config["statistics"] = statistics
        statistics["enabled"] = defaults["statistics"]["enabled"]
        statistics["retention_days"] = defaults["statistics"]["retention_days"]

        self._merge_pipeline_defaults(config, defaults["pipeline"])
        for model_id, keys in {
            PipelineStage.MOONDREAM: ("enabled", "prompt"),
            PipelineStage.OLLAMA:    ("enabled", "prompt"),
            PipelineStage.DEEPFACE:  ("enabled", "use_retinaface"),
            PipelineStage.FER:       ("enabled",),
        }.items():
            entry = self.get_pipeline_entry(config, model_id)
            if entry is None:
                continue
            for key in keys:
                entry[key] = copy.deepcopy(pipeline_defaults[model_id][key])

        return config

    def get_pipeline_entry(self, config, model_id):
        """
        Liefert einen Pipeline-Eintrag anhand seiner ID.
        :param config: Konfigurations-Dictionary.
        :param model_id: ID des gesuchten Eintrags (z.B. PipelineStage.DEEPFACE).
        :return: Pipeline-Eintrag als Dict, oder None wenn nicht gefunden.
        """
        pipeline = config.get("pipeline", [])
        if not isinstance(pipeline, list):
            return None
        return next(
            (entry for entry in pipeline if isinstance(entry, dict) and entry.get("id") == model_id),
            None
        )

    def _merge_dict_defaults(self, target, defaults, skip_keys=None):
        """
        Fügt fehlende Werte aus defaults rekursiv in target ein.
        Bestehende Werte in target werden nicht überschrieben.
        :param target: Ziel-Dictionary.
        :param defaults: Standardwerte-Dictionary.
        :param skip_keys: Schlüssel, die beim Merge übersprungen werden.
        """
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
        """
        Ergänzt fehlende Pipeline-Einträge und deren Felder.
        Neue Modelle in pipeline_defaults werden automatisch hinzugefügt,
        bestehende Einträge werden nur um fehlende Felder ergänzt.
        :param config: Konfigurations-Dictionary.
        :param pipeline_defaults: Liste mit Pipeline-Default-Einträgen.
        """
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
