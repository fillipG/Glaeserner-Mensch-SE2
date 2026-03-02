import os
import sys
import time
from datetime import datetime
import yaml
import cv2
from sympy import true, false

try:
    import torch
except ImportError:
    pass

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import QApplication


class YOLOWorker(QThread):
    def run(self):
        # Muss hier importiert werden wegen Konflikten
        global photo_taken
        from PersonPhotoCapture import PersonPhotoCapture

        # Ordner für gespeicherte Bilder erstellen
        os.makedirs("main_image", exist_ok=True)

        print("--- YOLO Worker: ACTIVE ---")
        try:
            # ==========================
            # Foto aufnehmen mit PersonPhotoCapture
            # ==========================
            photo_capture = PersonPhotoCapture(save_dir="main_image", photo_delay=3)
            photo_taken = False
            while not self.isInterruptionRequested() and not photo_taken:
                captured_frame = photo_capture.capture_photo()
                photo_taken = True
                # ==========================
                # Bild speichern
                # ==========================
                if captured_frame is not None:  # Prüfen, ob ein Bild aufgenommen wurde
                    timestamp = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")  # Zeitstempel erzeugen
                    filename = f"main_image/main_{timestamp}.jpg"
                    cv2.imwrite(filename, captured_frame)  # Bild speichern
                    print(f"Bild gespeichert: {filename}")  # Info ausgeben
                time.sleep(0.1)
        except Exception as e:
            print(f"YOLO Thread Error: {e}")

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

"""
PIPELINE WORKER KLASSE:
Dieser Worker läuft in einem eigenen Hintergrund-Thread (QThread).
Seine Aufgabe ist es, kontinuierlich den "final"-Ordner zu überwachen,
KI-Ergebnisse (YAML-Dateien) zusammenzuführen und die fertigen
Daten an die GUI zu senden.
"""
class PipelineWorker(QThread):
    # Signal, um Ergebnisse an die GUI zu senden
    result_ready = pyqtSignal(str, list)
    def run(self):
        # Import innerhalb des Threads, um Konflikte beim Start zu vermeiden
        from pipelinemanager import PipelineManager
        try:
            # Konfiguration laden
            with open("config.yaml", "r") as f:
                config_data = yaml.safe_load(f)

            # Initialisierung des Managers innerhalb des Threads
            self.manager = PipelineManager(config_data)

            # Das Signal des Managers mit dem Signal des Workers verknüpfen
            self.manager.data_finalized.connect(self.result_ready.emit)

            print("--- Pipeline Worker: ACTIVE ---")

            # Endlosschleife, solange der Thread nicht gestoppt wird
            while not self.isInterruptionRequested():
                # Ordner auf neue KI-Ergebnisse prüfen
                self.manager.check_for_updates()

                # Kurze Pause, um die CPU nicht zu überlasten
                time.sleep(0.5)

        except Exception as e:
            # Fehler abfangen, damit nicht die gesamte App abstürzt
            print(f"Pipeline Thread Error: {e}")


def run_app():
    """
    Haupteinstiegspunkt für die Anwendung.
    Initialisiert die GUI und startet die Hintergrund-Prozesse in einer sicheren Reihenfolge
    """
    from GUI import ScalingAkteGUI

    # Initialisierung von PyQt
    app = QApplication(sys.argv)
    window = ScalingAkteGUI()

    # 1. Zuerst die GUI anzeigen:
    # Kann auch maximiert gestartet werden
    window.show()
    #window.showMaximized()

    # processEvents zwingt Windows dazu, das Fenster sofort zu zeichnen,
    # bevor der Prozessor mit dem Laden der KI-Modelle beginnt
    app.processEvents()

    # 2. Pipeline-Worker starten:
    pipeline_thread = PipelineWorker()

    pipeline_thread.result_ready.connect(
        lambda id, data: window.handle_new_dataset(data),
        Qt.ConnectionType.QueuedConnection
    )
    # Referenz am Fenster-Objekt, damit der Python-Garbage-Collector den Thread nicht löscht, während er noch läuft
    window._pipeline = pipeline_thread

    # Eine kurze Pause gibt dem Betriebssystem Zeit, Ressourcen für den Thread bereitzustellen
    time.sleep(0.5)
    pipeline_thread.start()

    # 3. YOLO-Worker starten
    yolo_thread = YOLOWorker()
    window._yolo = yolo_thread

    # Kurze Verzögerung, um Hardware-Konflikte zu vermeiden
    time.sleep(0.5)
    yolo_thread.start()

    # app.exec() startet die Ereignisschleife der GUI und blockiert hier, bis das Fenster geschlossen wird.
    exit_code = app.exec()

    # Cleanup:
    # Wenn das Fenster geschlossen wird, senden wir ein Abbruch-Signal an die Threads,
    # damit diese ihre Schleifen sauber beenden und die Kamera/Dateizugriffe freigeben.
    pipeline_thread.requestInterruption()
    yolo_thread.requestInterruption()

    # Beendet das Programm mit dem entsprechenden Exit-Code
    sys.exit(exit_code)


if __name__ == "__main__":
    run_app()