import os
import time
from datetime import datetime
import yaml
import cv2
from pipelinemanager import PipelineManager


def run_yolo():
    os.makedirs("main_image", exist_ok=True)  # Erstellen des Ordners

    timestamp = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")

    camera = cv2.VideoCapture(0)  # Öffnen der Kamera
    ret, frame = camera.read()  # Aufnehmen

    # boolean ret: True, wenn das Bild erfolgreich aufgenommen wurde
    if ret:
        cv2.imwrite(f"main_image/main_{timestamp}.jpg", frame)
        print(f"Bild gespeichert: main_image/main_{timestamp}.jpg")
    else:
        print("Bild konnte nicht gelesen werden")


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