import math
import time
from datetime import datetime

import cv2
import yaml


class PersonPhotoCapture:
    """
    Kamera-Manager mit zwei Modi:
    1. capture_mode: voller Aufnahme-Modus mit Countdown und optionaler Vorschau
    2. presence_mode: schnelle Praesenzpruefung ohne Countdown oder Preview

    Ausserdem: take_photo() fuer sofortiges Einzelbild ohne Speicherung
    """

    def __init__(self, model, photo_delay=3, lost_tolerance=1.5):
        self.PHOTO_DELAY_SECONDS = photo_delay
        self.PERSON_LOST_TOLERANCE = lost_tolerance
        self._cap = None
        self.model = model
        self.language = "de"
        print("YOLO Modell uebernommen.")

    # ----------------------------
    # Runtime-Konfiguration
    # ----------------------------
    def update_runtime_config(self, photo_delay=None, language=None):
        """Aendert Foto-Delay und Sprache zur Laufzeit"""
        if photo_delay is not None:
            self.PHOTO_DELAY_SECONDS = int(photo_delay)
        if language in {"de", "en"}:
            self.language = language

    # ----------------------------
    # Kamerazugriff
    # ----------------------------
    def ensure_camera_open(self, log_open=True):
        """Oeffnet Kamera, falls nicht schon offen"""
        if self._cap is not None and self._cap.isOpened():
            return self._cap

        config = {}
        try:
            with open("config.yaml", "r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
        except Exception:
            config = {}

        for index in [0, 1, 2]:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            time.sleep(0.3)
            if cap.isOpened():
                camera_cfg = config.get("camera", {})
                try:
                    width = int(camera_cfg.get("width", 1280))
                except (TypeError, ValueError):
                    width = 1280
                try:
                    height = int(camera_cfg.get("height", 720))
                except (TypeError, ValueError):
                    height = 720
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, 30)
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if log_open:
                    print(f"[KAMERA] Geoeffnet (Index {index})")
                    print(f"[KAMERA] Aufloesung: {actual_width}x{actual_height}")
                self._cap = cap
                return self._cap
            cap.release()

        print("Keine Kamera gefunden!")
        return None

    def release_camera(self):
        """Schliesst die Kamera sauber"""
        if self._cap:
            self._cap.release()
            self._cap = None

    # ----------------------------
    # Frame lesen
    # ----------------------------
    def _read_frame(self, log_open=True):
        """Liest einen Frame von der Kamera"""
        cap = self.ensure_camera_open(log_open=log_open)
        if not cap:
            return None
        ret, frame = cap.read()
        if not ret:
            print("Fehler beim Lesen des Frames")
            self.release_camera()
            return None
        return frame

    # ----------------------------
    # Personenerkennung
    # ----------------------------
    def _detect_person(self, frame):
        """
        Prueft:
        - person_present: ist ueberhaupt eine Person sichtbar?
        - person_valid: frontal, stabil, Augen/Schultern gut sichtbar
        """
        frame_h, frame_w = frame.shape[:2]
        person_present, person_valid = False, False

        results = self.model(frame, conf=0.6, verbose=False)
        if not results or results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
            return person_present, person_valid

        person_present = True
        person = results[0].keypoints.xy[0]
        confs = results[0].keypoints.conf[0]
        if len(person) <= 16:
            return person_present, person_valid

        # Keypoints
        head, left_eye, right_eye = person[0], person[1], person[2]
        left_shoulder, right_shoulder = person[5], person[6]

        head_visible = 20 < head[0] < frame_w - 20 and 20 < head[1] < frame_h - 20
        eyes_confident = confs[1] > 0.5 and confs[2] > 0.5
        shoulders_confident = confs[5] > 0.5 and confs[6] > 0.5
        eyes_level = abs(left_eye[1] - right_eye[1]) < 20

        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        face_ratio_valid = False
        if shoulder_width > 0:
            eye_distance = abs(left_eye[0] - right_eye[0])
            face_ratio_valid = 0.2 < (eye_distance / shoulder_width) < 0.6

        if head_visible and eyes_confident and shoulders_confident and eyes_level and face_ratio_valid:
            person_valid = True

        return person_present, person_valid

    # ----------------------------
    # Preview Overlay (optional)
    # ----------------------------
    def _emit_capture_preview(self, frame, remaining, person_valid):
        display = frame.copy()
        h, w = display.shape[:2]

        # Countdown
        if remaining and remaining > 0:
            text = str(max(1, math.ceil(remaining)))
            font_scale, thickness = 6.0, 10
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.putText(
                display,
                text,
                ((w - text_w) // 2, (h + text_h) // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (30, 200, 255),
                thickness,
                cv2.LINE_AA,
            )
        # Hinweis, wenn Person nicht frontal
        elif not person_valid:
            hint = "Please look at the camera" if self.language == "en" else "Bitte in die Kamera schauen"
            font_scale, thickness = 1.0, 2
            (text_w, text_h), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_x, text_y = (w - text_w) // 2, h - 30
            cv2.rectangle(display, (text_x - 10, text_y - text_h - 8), (text_x + text_w + 10, text_y + 8), (20, 20, 20), -1)
            cv2.putText(display, hint, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (80, 220, 255), thickness)

        return display

    # ----------------------------
    # Capture-Modus
    # ----------------------------
    def capture_mode(self, frame_callback=None, stop_requested_getter=None, mode_active_getter=None):
        """
        Voller Aufnahme-Modus mit:
        - Live-Preview
        - Stabilitaetspruefung
        - Countdown
        Gibt das Foto per return zurueck.
        """
        photo_taken = False
        start_time = None
        person_stable_since = None
        last_person_seen = None
        last_reported = None
        absence_logged = False
        stable_time_required = 0.7

        while True:
            if stop_requested_getter and stop_requested_getter():
                break
            if mode_active_getter and not mode_active_getter():
                break

            frame = self._read_frame()
            if frame is None:
                break

            current_time = time.time()
            person_present, person_valid = self._detect_person(frame)

            if person_present:
                last_person_seen = current_time
                absence_logged = False

            # Stabilitaet pruefen
            if person_valid:
                if person_stable_since is None:
                    person_stable_since = current_time
                if (current_time - person_stable_since) >= stable_time_required and start_time is None:
                    start_time = current_time
            else:
                person_stable_since = None
                if last_person_seen and current_time - last_person_seen > self.PERSON_LOST_TOLERANCE:
                    start_time = None
                    last_person_seen = None
                    last_reported = None
                    absence_logged = False

            # Countdown
            remaining = None
            if start_time:
                elapsed = current_time - start_time
                remaining = self.PHOTO_DELAY_SECONDS - elapsed
                if remaining <= 0 and not photo_taken:
                    photo_taken = True
                    if frame_callback:
                        frame_callback(frame.copy())
                    return frame

            # Preview Callback
            if frame_callback:
                frame_callback(self._emit_capture_preview(frame, remaining, person_valid))

            time.sleep(0.01)

        return None

    # ----------------------------
    # Presence-Modus
    # ----------------------------
    def presence_mode(self, stop_requested_getter=None):
        """
        Prueft, ob eine Person vorhanden ist, ohne Countdown oder Preview.
        """
        if stop_requested_getter and stop_requested_getter():
            return None

        frame = self._read_frame(log_open=False)
        if frame is None:
            return False

        person_present, _ = self._detect_person(frame)
        return person_present

    # ----------------------------
    # Einfaches Sofort-Foto
    # ----------------------------
    def take_photo(self):
        """
        Macht sofort ein Foto und gibt es als numpy-Array zurueck.
        """
        frame = self._read_frame()
        if frame is None:
            print("Konnte kein Bild aufnehmen.")
            return None
        print("Foto aufgenommen.")
        return frame
