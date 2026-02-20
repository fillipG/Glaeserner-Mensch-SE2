import os
import time
from datetime import datetime
import yaml
import cv2
from pipelinemanager import PipelineManager
from PersonPhotoCapture import PersonPhotoCapture


def run_yolo():

    # Ordner für gespeicherte Bilder erstellen
    os.makedirs("main_image", exist_ok=True)

    # ==========================
    # Foto aufnehmen mit PersonPhotoCapture
    # ==========================
    photo_capture = PersonPhotoCapture(save_dir="main_image", photo_delay=3)
    captured_frame = photo_capture.capture_photo()  # Kamera-Stream + YOLO + Countdown

    # ==========================
    # Bild speichern
    # ==========================
    if captured_frame is not None:  # Prüfen, ob ein Bild aufgenommen wurde
        timestamp = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")  # Zeitstempel erzeugen
        filename = f"main_image/main_{timestamp}.jpg"
        cv2.imwrite(filename, captured_frame)  # Bild speichern
        print(f"Bild gespeichert: {filename}")  # Info ausgeben
    else:
        print("Kein Bild aufgenommen")  # Fehlerhinweis


def stream_video():
    camera = cv2.VideoCapture(0)  # Öffnen der Kamera

    while True:
        ret, frame = camera.read()  # Aufnehmen
        if not ret:
            print("Fehler beim Lesen des Videoframes")
            break

        cv2.imshow("Live Stream", frame)  # Anzeigen des Videoframes

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Beenden mit 'q'
            break

    camera.release()
    cv2.destroyAllWindows()


def run_pipeline():
    """Haupteinstiegspunkt: Lädt Konfiguration und startet den Pipeline-Loop."""
    try:
        with open("config.yaml", "r") as f:
            config_data = yaml.safe_load(f)
    except Exception as e:
        print(f"Config Error: {e}")
        return

    # Instanziierung der Manager-Klasse aus pipelinemanager.py
    manager = PipelineManager(config_data)

    try:
        while True:
            manager.check_for_updates()
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    run_yolo()
    # stream_video()
    run_pipeline()