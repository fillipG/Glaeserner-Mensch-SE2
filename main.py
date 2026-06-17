"""
Zentrales Steuerungsskript (Main)
---------------------------------
Dieses Skript ist der Haupteinstiegspunkt der Museumsanwendung.
Es initialisiert alle Komponenten, verknüpft die Signale und startet
die Qt-Eventschleife.

Zuständigkeiten:
1. Start und Konfiguration der PyQt6-GUI.
2. Startup-Cleanup der temporären Ordner (verhindert Altdaten-Probleme).
3. Verdrahtung der Signale zwischen GUI, YOLOWorker und PipelineWorker.
4. Starten der lokalen Worker-Prozesse (Ollama).
5. Sauberes Herunterfahren aller Threads beim Beenden.

Hinweis: YOLOWorker und PipelineWorker wurden nach workers/ ausgelagert,
um diese Datei übersichtlich zu halten.

AUTOREN: Dennis Penner, Florian Hoeft
"""

import os
import sys
import time
from ultralytics import YOLO

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMessageBox


def clear_directory(path):
    """
    Sorgt für Datenkonsistenz beim Programmstart.

    KI-Modelle kommunizieren über Ordner (Inboxes). Ohne dieses Cleanup
    würden Ergebnisse vom vorherigen Programmdurchlauf beim nächsten Start
    fälschlicherweise als neue Ergebnisse erkannt werden.
    Wird für alle Inbox- und Ergebnis-Ordner beim Start aufgerufen.
    """
    os.makedirs(path, exist_ok=True)
    for entry in os.listdir(path):
        entry_path = os.path.join(path, entry)
        try:
            if os.path.isfile(entry_path) or os.path.islink(entry_path):
                os.unlink(entry_path)
            elif os.path.isdir(entry_path):
                # Rekursives Löschen, falls eine KI Unterordner erstellt hat
                for root, dirs, files in os.walk(entry_path, topdown=False):
                    for file_name in files:
                        os.unlink(os.path.join(root, file_name))
                    for dir_name in dirs:
                        os.rmdir(os.path.join(root, dir_name))
                os.rmdir(entry_path)
        except Exception as exc:
            print(f"Startup-Cleanup konnte {entry_path} nicht loeschen: {exc}")


def run_app():
    """
    Hauptfunktion: Initialisiert die GUI, startet die Hintergrund-Worker
    und sorgt für die Vernetzung der Signale.
    """
    from gui.main_gui import ScalingAkteGUI
    from local_worker_manager import LocalWorkerManager
    from workers.yolo_worker import YOLOWorker
    from workers.pipeline_worker import PipelineWorker

    yolo_thread = YOLOWorker()
    yolo_thread.prepare()

    app = QApplication(sys.argv)
    window = ScalingAkteGUI()
    window.show()
    app.processEvents()

    # Alle KI-Kommunikationsordner beim Start leeren (Altdaten aus vorherigem
    # Durchlauf würden sonst fälschlicherweise als neue Ergebnisse erkannt).
    # Die Pfade kommen aus path_service → base_dir in config.yaml steuert sie alle.
    from path_service import get_paths
    paths = get_paths()
    for path_key in ("final", "main_image", "sketch_dir", "ollama_inbox", "deepface_inbox", "moondream_inbox"):
        clear_directory(str(paths[path_key]))

    # Verwaltung lokaler Subprozesse (z.B. Ollama-Server für die Texte)
    worker_manager = LocalWorkerManager()
    window._local_worker_manager = worker_manager
    try:
        worker_manager.start_ollama_worker()
    except RuntimeError as exc:
        error_message = (
            "Der lokale Ollama-Worker konnte nicht gestartet werden.\n\n"
            f"{exc}"
        )
        print(error_message)
        QMessageBox.critical(window, "Ollama-Start fehlgeschlagen", error_message)
        # Sicherstellen, dass alle Subprozesse beim Schließen beendet werden
    app.aboutToQuit.connect(worker_manager.stop_workers)

    # Start des Datenfluss-Monitorings
    pipeline_thread = PipelineWorker()
    pipeline_thread.result_ready.connect(
        window.handle_pipeline_result,
        Qt.ConnectionType.QueuedConnection,
    )
    window._pipeline = pipeline_thread
    time.sleep(0.5)
    pipeline_thread.start()

    window._yolo = yolo_thread
    yolo_thread.frame_ready.connect(window.on_camera_frame)
    yolo_thread.person_presence_changed.connect(window.on_person_presence_changed)
    yolo_thread.photo_done.connect(window.show_loading_indicator)
    yolo_thread.startup_error.connect(
        lambda message: QMessageBox.critical(window, "YOLO-Start fehlgeschlagen", message)
    )
    window.camera_prewarm_requested.connect(yolo_thread.request_camera_prewarm)
    window.folder_closed.connect(yolo_thread.start_capture_mode)
    window.presence_monitoring_requested.connect(yolo_thread.start_presence_monitoring)

    time.sleep(0.5)
    yolo_thread.start()

    # Start der Qt-Eventschleife
    exit_code = app.exec()

    # Sauberes Herunterfahren der Threads bei Programmende
    pipeline_thread.requestInterruption()
    yolo_thread.requestInterruption()
    if not pipeline_thread.wait(3000):
        print("Pipeline thread did not stop within 3 seconds.")
    if not yolo_thread.wait(3000):
        print("YOLO thread did not stop within 3 seconds.")
    worker_manager.stop_workers()
    sys.exit(exit_code)


if __name__ == "__main__":
    run_app()
