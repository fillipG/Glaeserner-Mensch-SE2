import time

import cv2
import yaml
from PyQt6.QtCore import QTimer

from path_service import get_paths

LIVE_DEEPFACE_TRANSLATIONS = {
    "de": {
        "Man": "Mann",
        "Woman": "Frau",
        "Mann": "Mann",
        "Frau": "Frau",
        "angry": "wuetend",
        "disgust": "angewidert",
        "disgusted": "angewidert",
        "fear": "aengstlich",
        "fearful": "aengstlich",
        "happy": "gluecklich",
        "sad": "traurig",
        "surprise": "ueberrascht",
        "surprised": "ueberrascht",
        "neutral": "neutral",
    },
    "en": {
        "Man": "Man",
        "Woman": "Woman",
        "Mann": "Man",
        "Frau": "Woman",
        "angry": "angry",
        "disgust": "disgusted",
        "disgusted": "disgusted",
        "fear": "fearful",
        "fearful": "fearful",
        "happy": "happy",
        "sad": "sad",
        "surprise": "surprised",
        "surprised": "surprised",
        "neutral": "neutral",
    },
}


class LiveDeepFaceService:
    def __init__(self, config: dict):
        """Verwaltet Trigger, Dateiaustausch und Overlay-Daten fuer Live-DeepFace."""
        self._config = config if isinstance(config, dict) else {}
        self._live_deepface_enabled = False
        self._live_deepface_interval = 3
        self._live_deepface_max_faces = 4
        self._live_deepface_last_trigger = 0.0
        self._live_deepface_labels = {}
        self._live_deepface_pending = {}
        self._live_deepface_retry_count = 0
        self.apply_config(self._config)

    @property
    def max_faces(self) -> int:
        """Liefert die aktuell konfigurierte Obergrenze fuer Live-Gesichter."""
        return self._live_deepface_max_faces

    def apply_config(self, config: dict) -> None:
        """Liest die Live-DeepFace-Werte aus der Config und setzt die Laufzeitfelder."""
        self._config = config if isinstance(config, dict) else {}
        cfg = self._config.get("live_deepface", {})
        if not isinstance(cfg, dict):
            cfg = {}
        self._live_deepface_enabled = bool(cfg.get("enabled", False))
        try:
            self._live_deepface_interval = max(1, int(cfg.get("interval_seconds", 3)))
        except (TypeError, ValueError):
            self._live_deepface_interval = 3
        try:
            self._live_deepface_max_faces = max(1, min(4, int(cfg.get("max_faces", 4))))
        except (TypeError, ValueError):
            self._live_deepface_max_faces = 4

    def set_enabled(self, enabled: bool, save_callback=None) -> None:
        """Aktiviert oder deaktiviert die Live-Analyse und speichert optional die Config."""
        self._live_deepface_enabled = bool(enabled)
        cfg = self._config.setdefault("live_deepface", {})
        if not isinstance(cfg, dict):
            cfg = {}
            self._config["live_deepface"] = cfg
        cfg["enabled"] = self._live_deepface_enabled
        if save_callback is not None:
            save_callback()
        if not self._live_deepface_enabled:
            self.reset()

    def set_interval(self, value, save_callback=None) -> None:
        """Setzt das Trigger-Intervall neu und startet den Zeitzaehler sauber neu."""
        interval = max(1, int(value))
        self._live_deepface_interval = interval
        cfg = self._config.setdefault("live_deepface", {})
        if not isinstance(cfg, dict):
            cfg = {}
            self._config["live_deepface"] = cfg
        cfg["interval_seconds"] = interval
        self._live_deepface_last_trigger = 0.0
        if save_callback is not None:
            save_callback()

    def maybe_trigger(self, frame, faces, is_open: bool, is_animating: bool, loading_active: bool) -> None:
        """Triggert nur im passenden GUI-Zustand eine neue Live-DeepFace-Anfrage."""
        if (
            not self._live_deepface_enabled
            or is_open
            or is_animating
            or loading_active
        ):
            return

        now = time.time()
        if now - self._live_deepface_last_trigger < self._live_deepface_interval:
            return

        self._live_deepface_last_trigger = now
        self._trigger_live_deepface(frame, faces)

    def _trigger_live_deepface(self, frame, faces) -> None:
        """Schreibt aktuelle Gesichts-Crops in die DeepFace-Inbox und startet das Polling."""
        paths = get_paths()
        deepface_inbox = paths["deepface_inbox"]
        final_dir = paths["final"]

        try:
            deepface_inbox.mkdir(parents=True, exist_ok=True)
            final_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return

        try:
            for file_path in deepface_inbox.glob("live_face*.jpg"):
                file_path.unlink(missing_ok=True)
        except Exception:
            pass

        self._live_deepface_pending = {}
        self._live_deepface_retry_count = 0
        limited_faces = list((faces or [])[:self._live_deepface_max_faces])

        frame_h, frame_w = frame.shape[:2]
        for face_index, (x, y, w, h) in enumerate(limited_faces, start=1):
            x = max(0, int(x))
            y = max(0, int(y))
            w = max(0, int(w))
            h = max(0, int(h))
            x2 = min(frame_w, x + w)
            y2 = min(frame_h, y + h)
            crop = frame[y:y2, x:x2]
            if crop.size == 0:
                continue

            inbox_path = deepface_inbox / f"live_face{face_index}.jpg"
            output_path = final_dir / f"live_face{face_index}_deepface.yaml"
            try:
                output_path.unlink(missing_ok=True)
                if cv2.imwrite(str(inbox_path), crop):
                    self._live_deepface_pending[face_index] = output_path
            except Exception:
                pass

        if self._live_deepface_pending:
            QTimer.singleShot(500, self.collect_results)

    def collect_results(self) -> None:
        """Pollt fertige DeepFace-YAMLs und aktualisiert daraus die gespeicherten Labels."""
        if not self._live_deepface_pending:
            return
        max_retries = 6
        still_pending = {}

        for face_index, output_path in self._live_deepface_pending.items():
            if output_path.exists():
                try:
                    with open(output_path, "r", encoding="utf-8") as handle:
                        data = yaml.safe_load(handle) or {}
                    self._live_deepface_labels[face_index] = {
                        "alter": data.get("Alter", "?"),
                        "geschlecht": data.get("Geschlecht", "?"),
                        "emotion": data.get("Emotion", "?"),
                    }
                    output_path.unlink(missing_ok=True)
                except Exception:
                    still_pending[face_index] = output_path
            else:
                still_pending[face_index] = output_path

        if still_pending:
            if self._live_deepface_retry_count < max_retries:
                self._live_deepface_retry_count += 1
                self._live_deepface_pending = still_pending
                QTimer.singleShot(500, self.collect_results)
            else:
                self._live_deepface_pending = {}
                self._live_deepface_retry_count = 0
        else:
            self._live_deepface_pending = {}
            self._live_deepface_retry_count = 0

    def _translate_overlay_value(self, value, language: str) -> str:
        translated = LIVE_DEEPFACE_TRANSLATIONS.get(language, {}).get(value)
        if translated is not None:
            return translated
        return str(value)

    def get_overlay_data(self, faces: list, language: str = "de") -> list:
        """Nimmt aktuelle Gesichtsboxen entgegen und liefert sprachabhaengige Overlay-Texte zurueck."""
        overlays = []
        for face_index, (x, y, w, h) in enumerate((faces or [])[:self._live_deepface_max_faces], start=1):
            label_data = self._live_deepface_labels.get(face_index)
            if not label_data:
                continue
            gender = self._translate_overlay_value(label_data["geschlecht"], language)
            emotion = self._translate_overlay_value(label_data["emotion"], language)
            overlays.append({
                "text": f"{gender}, {label_data['alter']}, {emotion}",
                "x": int(x),
                "y": int(y),
            })
        return overlays

    def reset(self) -> None:
        """Setzt nur den internen Live-Zustand zurueck und laesst Dateien auf Platte unberuehrt."""
        self._live_deepface_labels = {}
        self._live_deepface_pending = {}
        self._live_deepface_retry_count = 0
        self._live_deepface_last_trigger = 0.0
