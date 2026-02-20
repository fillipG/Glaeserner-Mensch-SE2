import os
import time
from datetime import datetime
import cv2
from ultralytics import YOLO

class PersonPhotoCapture:
    """
    Automatische Fotoaufnahme bei erkannter Person.
    - Person muss vollständig sichtbar sein
    - Countdown vor Aufnahme
    - Stabiler Timer, falls Person kurz aus dem Bild geht
    """
    def __init__(self, save_dir="main_image", photo_delay=3, lost_tolerance=0.5):
        self.save_dir = save_dir
        self.PHOTO_DELAY_SECONDS = photo_delay           # Sekunden bis Foto
        self.PERSON_LOST_TOLERANCE = lost_tolerance     # Toleranz, falls Person kurz verschwindet

        os.makedirs(self.save_dir, exist_ok=True)       # Speicherordner erstellen
        self.model = YOLO("yolov8n-pose.pt")           # YOLO Pose-Modell laden

    def capture_photo(self):
        """
        Kamera öffnen, Person erkennen, Countdown anzeigen
        und aufgenommenen Frame zurückgeben.
        """
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)       # Kamera öffnen
        photo_taken = False                             # Flag, ob Foto aufgenommen
        start_time = None                               # Countdown-Startzeit
        last_person_seen = None

        while True:
            ret, frame = cap.read()                     # Frame aus Kamera lesen
            if not ret:
                print("Fehler beim Lesen des Frames")
                break

            results = self.model(frame, verbose=False) # YOLO-Erkennung
            yolo_img = results[0].plot()               # Keypoints visualisieren
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
                    # Prüfen, ob Kopf und Füße im Bild sind
                    if (head[0] > margin and head[0] < frame_w - margin and
                        head[1] > margin and head[1] < frame_h - margin and
                        left_foot[1] > margin and left_foot[1] < frame_h - margin and
                        right_foot[1] > margin and right_foot[1] < frame_h - margin):
                        person_complete = True
                        last_person_seen = current_time

            # ================== TIMER UND COUNTDOWN ==================
            if person_complete:
                if start_time is None:
                    start_time = current_time                # Countdown starten
            else:
                # Kurzzeitiges Verlassen des Bildes erlaubt
                if last_person_seen is not None:
                    if current_time - last_person_seen > self.PERSON_LOST_TOLERANCE:
                        start_time = None
                        photo_taken = False
                        last_person_seen = None

            # Countdown anzeigen & Foto aufnehmen
            if start_time is not None:
                elapsed = current_time - start_time
                remaining = self.PHOTO_DELAY_SECONDS - elapsed

                if remaining > 0:
                    cv2.putText(yolo_img, f"Foto in: {int(remaining)+1}", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                elif not photo_taken:
                    photo_taken = True
                    cap.release()
                    cv2.destroyAllWindows()
                    return frame  # Bild zurückgeben

            # Live-Frame anzeigen
            cv2.imshow("YOLO Pose - Fotoaufnahme", yolo_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):          # Abbrechen mit 'q'
                break

        cap.release()
        cv2.destroyAllWindows()
        return None