import os
import time
from datetime import datetime
import cv2
from ultralytics import YOLO

class PersonPhotoCapture:
    """
    Automatische Fotoaufnahme bei erkannter Person.
    - Person muss vollständig sichtbar sein
    - Countdown vor Aufnahme in der Konsole
    - Kein CV2-Fenster
    """
    def __init__(self, save_dir="main_image", photo_delay=3, lost_tolerance=0.5):
        self.save_dir = save_dir
        self.PHOTO_DELAY_SECONDS = photo_delay       # Sekunden bis Foto
        self.PERSON_LOST_TOLERANCE = lost_tolerance # Toleranz, falls Person kurz verschwindet

        self.model = YOLO("yolov8n-pose.pt")        # YOLO Pose-Modell laden

    def capture_photo(self):
        """
        Kamera öffnen, Person erkennen, Countdown in Konsole anzeigen
        und aufgenommenen Frame zurückgeben. Kein Fenster.
        """
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # Kamera öffnen
        photo_taken = False                         # Flag, ob Foto aufgenommen
        start_time = None                           # Countdown-Startzeit
        last_person_seen = None
        last_reported = None                        # letzte Sekunde, die ausgegeben wurde

        while True:
            ret, frame = cap.read()                 # Frame aus Kamera lesen
            if not ret:
                print("Fehler beim Lesen des Frames")
                break

            results = self.model(frame, verbose=False) # YOLO-Erkennung
            r = results[0]

            current_time = time.time()
            frame_h, frame_w = frame.shape[:2]

            # ================== PERSON VOLLSTÄNDIG ERKENNEN ==================
            person_complete = False
            if r.keypoints is not None and len(r.keypoints.xy) > 0:
                person = r.keypoints.xy[0]
                if len(person) > 16:
                    head = person[0]
                    left_foot = person[15]
                    right_foot = person[16]
                    margin = 20
                    # Prüfen, ob Kopf im Bild ist
                    if (head[0] > margin and head[0] < frame_w - margin and
                        head[1] > margin and head[1] < frame_h - margin):
                        person_complete = True
                        last_person_seen = current_time

            # ================== TIMER LOGIK ==================
            if person_complete:
                if start_time is None:
                    start_time = current_time           # Countdown starten
            else:
                # Kurzzeitiges Verlassen erlaubt
                if last_person_seen is not None:
                    if current_time - last_person_seen > self.PERSON_LOST_TOLERANCE:
                        start_time = None
                        photo_taken = False
                        last_person_seen = None
                        last_reported = None

            # Countdown und Fotoaufnahme
            if start_time is not None:
                elapsed = current_time - start_time
                remaining = self.PHOTO_DELAY_SECONDS - elapsed

                # Countdown in Konsole ausgeben
                sec_remaining = int(remaining) + 1
                if remaining > 0 and sec_remaining != last_reported:
                    print(f"Foto in: {sec_remaining} Sekunden...")
                    last_reported = sec_remaining

                elif remaining <= 0 and not photo_taken:
                    photo_taken = True
                    print("FOTO AUFGENOMMEN!")
                    cap.release()
                    return frame  # Bild zurückgeben

            # kurze Pause, um CPU zu schonen
            time.sleep(0.01)

        cap.release()
        return None