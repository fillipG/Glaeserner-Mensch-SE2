import os
import time
from datetime import datetime
import cv2
from ultralytics import YOLO


class PersonPhotoCapture:
    """
    Klasse zur automatischen Fotoaufnahme bei erkannter Person.

    Ziel:
    - Foto wird nur aufgenommen, wenn eine Person
      stabil und frontal zur Kamera steht.
    - Rücken- oder Seitenansichten werden ausgeschlossen.
    - Kurze Tracking-Aussetzer führen NICHT sofort zum Reset.

    Technische Kernpunkte:
    - YOLOv8 Pose-Modell zur Keypoint-Erkennung
    - Geometrische Prüfung der Pose (Augen + Schultern)
    - Stabilitäts- und Toleranzlogik
    """

    def __init__(self, save_dir="main_image", photo_delay=3, lost_tolerance=1.5):
        """
        Initialisiert das Aufnahme-System.

        :param save_dir: Zielverzeichnis (aktuell nicht genutzt, aber vorbereitet)
        :param photo_delay: Sekunden bis zur Fotoaufnahme (Countdown)
        :param lost_tolerance: Toleranzzeit bei Tracking-Verlust
        """

        self.save_dir = save_dir
        self.PHOTO_DELAY_SECONDS = photo_delay
        self.PERSON_LOST_TOLERANCE = lost_tolerance

        # Laden des YOLO Pose Modells
        # Das Modell erkennt Körper-Keypoints (Augen, Schultern, etc.)
        print("Lade YOLO Modell...")
        self.model = YOLO("yolov8n-pose.pt")
        print("YOLO Modell geladen.")

    # =========================================================
    # KAMERA INITIALISIERUNG UND DEBUGGING
    # =========================================================
    def open_camera(self):
        """
        Versucht, eine verfügbare Kamera zu öffnen.
        Es werden mehrere Indizes getestet (0–2),
        um unterschiedliche Geräte-Konfigurationen abzudecken.
        """

        print("Versuche Kamera zu öffnen...")

        for index in [0, 1, 2]:
            print(f"Teste Kamera Index {index}...")

            # DirectShow Backend für Windows
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            time.sleep(0.5)

            print(f"isOpened (Index {index}):", cap.isOpened())

            if cap.isOpened():
                print(f"[{datetime.now()}] ✅ Kamera erfolgreich geöffnet! (Index {index})")

                # Debug-Informationen zur Auflösung
                print("Frame Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                print("Frame Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                return cap
            else:
                print(f"[{datetime.now()}] ❌ Kamera Index {index} nicht verfügbar.")
                cap.release()

        print("❌ Keine Kamera gefunden!")
        return None

    # =========================================================
    # HAUPTFUNKTION ZUR FOTOAUFNAHME
    # =========================================================
    def capture_photo(self):
        """
        Hauptlogik:
        - Kamera öffnen
        - Person frontal erkennen
        - Stabilität prüfen
        - Countdown starten
        - Foto aufnehmen
        """

        cap = self.open_camera()
        if cap is None:
            return None

        # Statusvariablen
        photo_taken = False
        start_time = None
        person_stable_since = None
        last_person_seen = None
        last_reported = None

        # Stabilitäts- und Toleranzparameter
        STABLE_TIME_REQUIRED = 0.7
        MAX_LOST_TIME = self.PERSON_LOST_TOLERANCE

        print("Starte Personenerkennung...")

        while True:
            ret, frame = cap.read()

            # Fehlerbehandlung bei Kameraproblem
            if not ret:
                print("Fehler beim Lesen des Frames")
                break

            # YOLO-Inferenz (Pose-Erkennung)
            results = self.model(frame, conf=0.6, verbose=False)
            r = results[0]

            current_time = time.time()
            frame_h, frame_w = frame.shape[:2]

            person_valid = False

            # =========================================================
            # FRONTAL-ERKENNUNG
            # =========================================================
            # Ziel: Rücken- oder Seitenansichten ausschließen
            # Methode: Geometrische Prüfung von Augen + Schultern
            # =========================================================
            if r.keypoints is not None and len(r.keypoints.xy) > 0:

                person = r.keypoints.xy[0]
                confs = r.keypoints.conf[0]

                if len(person) > 16:

                    # Relevante Keypoints
                    head = person[0]
                    left_eye = person[1]
                    right_eye = person[2]
                    left_shoulder = person[5]
                    right_shoulder = person[6]

                    # Confidence-Werte der Keypoints
                    head_conf = confs[0]
                    left_eye_conf = confs[1]
                    right_eye_conf = confs[2]
                    left_shoulder_conf = confs[5]
                    right_shoulder_conf = confs[6]

                    margin = 20

                    # 1. Kopf muss vollständig im Bild sein
                    head_visible = (
                        head[0] > margin and head[0] < frame_w - margin and
                        head[1] > margin and head[1] < frame_h - margin
                    )

                    # 2. Augen müssen zuverlässig erkannt sein
                    eyes_confident = (
                        left_eye_conf > 0.5 and
                        right_eye_conf > 0.5
                    )

                    # 3. Schultern müssen zuverlässig erkannt sein
                    shoulders_confident = (
                        left_shoulder_conf > 0.5 and
                        right_shoulder_conf > 0.5
                    )

                    # 4. Augenabstand berechnen
                    eye_distance = abs(left_eye[0] - right_eye[0])

                    # 5. Schulterbreite berechnen
                    shoulder_width = abs(left_shoulder[0] - right_shoulder[0])

                    face_ratio_valid = False
                    if shoulder_width > 0:
                        ratio = eye_distance / shoulder_width

                        # Typisches Verhältnis bei Frontalansicht
                        if 0.2 < ratio < 0.6:
                            face_ratio_valid = True

                    # 6. Augen sollten ungefähr auf gleicher Höhe sein
                    eyes_level = abs(left_eye[1] - right_eye[1]) < 20

                    # Finaler Frontal-Check
                    if (head_visible and
                        eyes_confident and
                        shoulders_confident and
                        eyes_level and
                        face_ratio_valid):

                        person_valid = True
                        last_person_seen = current_time

            # =========================================================
            # STABILITÄTSLOGIK
            # =========================================================
            # Countdown startet erst,
            # wenn die Person über eine gewisse Zeit stabil erkannt wird.
            # =========================================================
            if person_valid:

                if person_stable_since is None:
                    person_stable_since = current_time

                # Stabilitätsprüfung
                if (current_time - person_stable_since) >= STABLE_TIME_REQUIRED:
                    if start_time is None:
                        print("Person stabil frontal erkannt → Countdown startet")
                        start_time = current_time
            else:
                # Reset nur, wenn Person länger als Toleranzzeit weg ist
                if last_person_seen is not None:
                    if current_time - last_person_seen > MAX_LOST_TIME:
                        print("Person zu lange verloren → Reset")

                        start_time = None
                        person_stable_since = None
                        photo_taken = False
                        last_person_seen = None
                        last_reported = None

            # =========================================================
            # COUNTDOWN-LOGIK
            # =========================================================
            if start_time is not None:

                elapsed = current_time - start_time
                remaining = self.PHOTO_DELAY_SECONDS - elapsed
                sec_remaining = int(remaining) + 1

                # Countdown-Ausgabe nur bei Änderung
                if remaining > 0 and sec_remaining != last_reported:
                    print(f"Foto in: {sec_remaining} Sekunden...")
                    last_reported = sec_remaining

                # Foto aufnehmen
                elif remaining <= 0 and not photo_taken:
                    photo_taken = True
                    print(f"[{datetime.now()}] 📸 FOTO AUFGENOMMEN!")

                    cap.release()
                    return frame

            # Kurze Pause zur CPU-Entlastung
            time.sleep(0.01)

        cap.release()
        print("Kamera geschlossen.")
        return None


# =========================================================
# TESTBLOCK (Direkter Skriptstart)
# =========================================================
if __name__ == "__main__":

    capture = PersonPhotoCapture(photo_delay=3)
    image = capture.capture_photo()

    if image is not None:
        print("Bild erfolgreich aufgenommen.")
    else:
        print("Kein Bild aufgenommen.")