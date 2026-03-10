import time
from datetime import datetime

import cv2
from ultralytics import YOLO


class PersonPhotoCapture:
    """
    Teilt Kameraarbeit in zwei Modi:
    - capture_mode: volle Foto-Logik mit Countdown und Preview
    - presence_mode: einzelne, leichte Praesenzpruefung ohne Preview
    """

    def __init__(self, photo_delay=3, lost_tolerance=1.5):
        self.PHOTO_DELAY_SECONDS = photo_delay
        self.PERSON_LOST_TOLERANCE = lost_tolerance
        self._cap = None

        print("Lade YOLO Modell...")
        self.model = YOLO("yolov8n-pose.pt")
        print("YOLO Modell geladen.")

    # Runtime-Werte koennen vom Worker vor jedem Zyklus nachgeladen werden.
    def update_runtime_config(self, photo_delay=None):
        if photo_delay is not None:
            self.PHOTO_DELAY_SECONDS = int(photo_delay)

    # Die Kamera bleibt pro Modus wiederverwendbar offen, bis sie explizit freigegeben wird.
    def ensure_camera_open(self):
        if self._cap is not None and self._cap.isOpened():
            return self._cap

        print("Versuche Kamera zu oeffnen...")
        for index in [0, 1, 2]:
            print(f"Teste Kamera Index {index}...")
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            time.sleep(0.3)

            print(f"isOpened (Index {index}):", cap.isOpened())
            if cap.isOpened():
                print(f"[{datetime.now()}] Kamera erfolgreich geoeffnet! (Index {index})")
                print("Frame Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                print("Frame Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self._cap = cap
                return self._cap

            print(f"[{datetime.now()}] Kamera Index {index} nicht verfuegbar.")
            cap.release()

        print("Keine Kamera gefunden!")
        return None

    def release_camera(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # Gemeinsamer Frame-Leser fuer Capture- und Presence-Modus.
    def _read_frame(self):
        cap = self.ensure_camera_open()
        if cap is None:
            return None

        ret, frame = cap.read()
        if not ret:
            print("Fehler beim Lesen des Frames")
            self.release_camera()
            return None
        return frame

    # Trennt zwischen "Person ist da" und "Person ist frontal genug fuer den Countdown".
    def _detect_person(self, frame):
        frame_h, frame_w = frame.shape[:2]
        person_present = False
        person_valid = False

        results = self.model(frame, conf=0.6, verbose=False)
        result = results[0]
        if result.keypoints is None or len(result.keypoints.xy) == 0:
            return person_present, person_valid

        person_present = True
        person = result.keypoints.xy[0]
        confs = result.keypoints.conf[0]
        if len(person) <= 16:
            return person_present, person_valid

        head = person[0]
        left_eye = person[1]
        right_eye = person[2]
        left_shoulder = person[5]
        right_shoulder = person[6]

        head_visible = (
            head[0] > 20 and head[0] < frame_w - 20 and
            head[1] > 20 and head[1] < frame_h - 20
        )
        eyes_confident = confs[1] > 0.5 and confs[2] > 0.5
        shoulders_confident = confs[5] > 0.5 and confs[6] > 0.5
        eyes_level = abs(left_eye[1] - right_eye[1]) < 20

        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        face_ratio_valid = False
        if shoulder_width > 0:
            eye_distance = abs(left_eye[0] - right_eye[0])
            ratio = eye_distance / shoulder_width
            face_ratio_valid = 0.2 < ratio < 0.6

        if head_visible and eyes_confident and shoulders_confident and eyes_level and face_ratio_valid:
            person_valid = True

        return person_present, person_valid

    # Preview-Overlays werden nur im Capture-Modus gezeichnet.
    def _emit_capture_preview(self, frame, remaining, person_valid):
        display = frame.copy()
        height, width = display.shape[:2]

        if remaining is not None and remaining > 0:
            sec_remaining = int(remaining) + 1
            text = str(sec_remaining)
            font_scale, thickness = 6.0, 10
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            center_x = (width - text_w) // 2
            center_y = (height + text_h) // 2
            cv2.putText(
                display,
                text,
                (center_x + 4, center_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness + 4,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                text,
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (30, 200, 255),
                thickness,
                cv2.LINE_AA,
            )
        elif not person_valid:
            hint = "Bitte in die Kamera schauen"
            font_scale, thickness = 1.0, 2
            (text_w, text_h), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_x = (width - text_w) // 2
            text_y = height - 30
            cv2.rectangle(
                display,
                (text_x - 10, text_y - text_h - 8),
                (text_x + text_w + 10, text_y + 8),
                (20, 20, 20),
                -1,
            )
            cv2.putText(
                display,
                hint,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (80, 220, 255),
                thickness,
                cv2.LINE_AA,
            )

        return display

    def capture_mode(self, frame_callback=None, stop_requested_getter=None, mode_active_getter=None):
        """
        Voller Aufnahme-Modus mit Live-Preview, Stabilitaetspruefung und Foto-Countdown.
        """
        photo_taken = False
        start_time = None
        person_stable_since = None
        last_person_seen = None
        last_reported = None
        absence_logged = False
        stable_time_required = 0.7

        print("Starte Personenerkennung...")

        while True:
            if stop_requested_getter is not None and stop_requested_getter():
                break
            if mode_active_getter is not None and not mode_active_getter():
                break

            frame = self._read_frame()
            if frame is None:
                break

            current_time = time.time()
            person_present, person_valid = self._detect_person(frame)

            if person_present:
                last_person_seen = current_time
                absence_logged = False

            # Erst stabile Frontalerkennung startet den eigentlichen Foto-Countdown.
            if person_valid:
                if person_stable_since is None:
                    person_stable_since = current_time
                if (current_time - person_stable_since) >= stable_time_required and start_time is None:
                    print("Person stabil frontal erkannt -> Countdown startet")
                    start_time = current_time
            else:
                person_stable_since = None
                if last_person_seen is not None:
                    time_since_seen = current_time - last_person_seen
                    if not person_present and not absence_logged and time_since_seen <= self.PERSON_LOST_TOLERANCE:
                        remaining_tolerance = max(0.0, self.PERSON_LOST_TOLERANCE - time_since_seen)
                        print(
                            f"Person kurz verloren -> warte noch {remaining_tolerance:.1f}s "
                            f"bis Countdown-Reset"
                        )
                        absence_logged = True
                    if not person_present and time_since_seen > self.PERSON_LOST_TOLERANCE:
                        print("Person zu lange verloren -> Countdown wird zurueckgesetzt")
                        start_time = None
                        last_person_seen = None
                        last_reported = None
                        absence_logged = False

            remaining = None
            # Der Countdown laeuft erst nach stabiler Freigabe und endet direkt mit dem Foto-Frame.
            if start_time is not None:
                elapsed = current_time - start_time
                remaining = self.PHOTO_DELAY_SECONDS - elapsed
                sec_remaining = int(remaining) + 1
                if remaining > 0 and sec_remaining != last_reported:
                    print(f"Foto in: {sec_remaining} Sekunden...")
                    last_reported = sec_remaining
                elif remaining <= 0 and not photo_taken:
                    photo_taken = True
                    if frame_callback is not None:
                        try:
                            # Das letzte Preview-Bild fuer die Analyse bleibt ohne Countdown-Overlay stehen.
                            frame_callback(frame.copy())
                        except Exception:
                            pass
                    print(f"[{datetime.now()}] FOTO AUFGENOMMEN!")
                    return frame

            if frame_callback is not None:
                try:
                    frame_callback(self._emit_capture_preview(frame, remaining, person_valid))
                except Exception:
                    pass

            time.sleep(0.01)

        return None

    # Presence-Modus prueft nur einmal, ob noch jemand da ist, ohne Preview oder Countdown.
    def presence_mode(self, stop_requested_getter=None):
        """
        Leichte Praesenzpruefung fuer Auto-Close. Kein Countdown, kein Preview.
        """
        if stop_requested_getter is not None and stop_requested_getter():
            return None

        frame = self._read_frame()
        if frame is None:
            return False

        person_present, _ = self._detect_person(frame)
        return person_present
