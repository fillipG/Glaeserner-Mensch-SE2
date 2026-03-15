"""
Zentrales Steuerungsskript (Main)
---------------------------------
Dieses Skript dient als Haupteinstiegspunkt. Es verwaltet
die grafische Benutzeroberfläche und koordiniert die im Hintergrund laufenden
Prozesse für die Bildaufnahme und die KI-Pipeline.

Zuständigkeiten:
1. Start und Konfiguration der PyQt6-GUI.
2. Steuerung des YOLOWorkers (Kamera-Input und Personenerkennung).
3. Steuerung des PipelineWorkers (Überwachung der KI-Ergebnisse in den Verzeichnissen).
4. Vorbereitung der Ordnerstruktur und Bereinigung alter Daten beim Programmstart.
5. Koordinierung eines sauberen Programmendes.

AUTOREN: Dennis Penner, Florian Hoeft
"""

import os
import sys
import time
from enum import Enum
from threading import Lock
from ultralytics import YOLO

import yaml
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMessageBox


def clear_directory(path):
    """
    Sorgt für Datenkonsistenz beim Programmstart.
    KI-Modelle kommunizieren über Ordner (Inboxes). Um zu verhindern, dass
    Ergebnisse vom vorherigen Programmdurchlauf fälschlicherweise als neu
    erkannt werden, löscht diese Funktion alle Inhalte in den temporären Ordnern.
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


class WorkerState(Enum):
    """
    Status-Definitionen für die Kamera-Logik.
    Verhindert Konflikte, indem klar geregelt ist, ob die Kamera gerade
    nach Personen sucht, ein Foto schießt oder nur die Anwesenheit prüft.
    """
    IDLE = "idle"  # Wartet auf eine Person
    ANALYZING = "analyzing"  # Foto wurde geschossen, KIs arbeiten
    PRESENCE_MONITORING = "presence_monitoring"  # Prüft, ob Nutzer noch da ist


class YOLOWorker(QThread):
    """
    Der YOLOWorker kapselt die gesamte Bildverarbeitung der Kamera.
    Er nutzt YOLOv8-Pose, um Personen im Sichtfeld zu erkennen und den
    automatisierten Foto-Prozess (Countdown) einzuleiten.
    """
    frame_ready = pyqtSignal(object)  # Sendet Live-Bilder an die GUI
    photo_done = pyqtSignal()  # Signalisiert: Foto erfolgreich erstellt
    person_presence_changed = pyqtSignal(bool)  # Meldet, ob Person den Platz verlassen hat
    startup_error = pyqtSignal(str)  # Meldet Fehler bei der Hardware-Initialisierung

    def __init__(self):
        super().__init__()
        self._state = WorkerState.IDLE
        self._presence_check_interval_ms = 2000
        self._photo_capture = None
        self._startup_error_message = None

    def _get_state(self):
        return self._state

    def _set_state(self, state):
        self._state = state

    def start_capture_mode(self):
        """Versetzt den Worker zurück in den Suchmodus (z.B. nach einem Reset)."""
        print("YOLO Worker: IDLE")
        self._set_state(WorkerState.IDLE)

    def start_analyzing_mode(self):
        """Stoppt die Kamera-Aktionen, während die KIs im Hintergrund rechnen."""
        print("YOLO Worker: ANALYZING")
        self._set_state(WorkerState.ANALYZING)

    def start_presence_monitoring(self):
        """Aktiviert die Prüfung, ob die Person nach dem Prozess noch vor dem Gerät steht."""
        print("YOLO Worker: PRESENCE_MONITORING")
        self._set_state(WorkerState.PRESENCE_MONITORING)

    def prepare(self):
        """
        Lädt das YOLO-Modell vorab.
        """
        if self._photo_capture is not None or self._startup_error_message is not None:
            return

        try:
            from PersonPhotoCapture import PersonPhotoCapture

            os.makedirs("General ordner/main_image", exist_ok=True)
            # Laden des vortrainierten YOLOv8-Pose-Modells für Personenerkennung
            print("YOLO Worker: bereite Modell im Hauptthread vor...")
            model = YOLO("yolov8n-pose.pt")
            self._photo_capture = PersonPhotoCapture(model=model, photo_delay=3)
        except Exception as exc:
            self._startup_error_message = (
                "YOLO/Torch konnte nicht initialisiert werden.\n"
                "Wahrscheinlich fehlt auf diesem Windows-System eine Torch-Abhaengigkeit "
                "oder es ist eine unpassende Torch-Installation aktiv.\n"
                f"Details: {exc}"
            )

    def run(self):
        """
        Hauptschleife des Kamera-Threads.
        Reagiert dynamisch auf Zustandsänderungen und Konfigurationsanpassungen.
        """
        self.prepare()
        if self._startup_error_message is not None:
            print(self._startup_error_message)
            self.startup_error.emit(self._startup_error_message)
            return

        photo_capture = self._photo_capture
        print("--- YOLO Worker: ACTIVE ---")

        try:
            while not self.isInterruptionRequested():
                # Ermöglicht das Ändern des Countdowns im laufenden Betrieb via config.yaml
                with open("config.yaml", "r", encoding="utf-8") as handle:
                    cfg = yaml.safe_load(handle) or {}

                photo_delay = int(cfg.get("photo_delay") or 3)
                self._presence_check_interval_ms = max(
                    1000,
                    int(cfg.get("no_person_check_interval_ms", 2000) or 2000),
                )
                photo_capture.update_runtime_config(photo_delay=photo_delay)

                state = self._get_state()

                # FALL 1: Suche nach Personen & Automatischer Snapshot
                if state == WorkerState.IDLE:
                    captured_frame = photo_capture.capture_mode(
                        frame_callback=lambda frame: self.frame_ready.emit(frame),
                        stop_requested_getter=self.isInterruptionRequested,
                        mode_active_getter=lambda: self._get_state() == WorkerState.IDLE,
                    )
                    if captured_frame is None:
                        time.sleep(0.05)
                        continue

                    import cv2
                    # Speichert das Bild zentral ab, damit die KI-Docker-Container darauf zugreifen können
                    filename = "General ordner/main_image/face_trigger.jpg"
                    cv2.imwrite(filename, captured_frame)
                    print(f"Bild gespeichert: {filename}")
                    photo_capture.release_camera()
                    self.start_analyzing_mode()
                    self.photo_done.emit()
                    continue

                # FALL 2: Wartemodus während der Analyse (Ressourcenschonung)
                if state == WorkerState.ANALYZING:
                    photo_capture.release_camera()
                    time.sleep(0.1)
                    continue

                # FALL 3: Prüfen, ob die Person den Erfassungsbereich verlassen hat
                if state == WorkerState.PRESENCE_MONITORING:
                    is_present = photo_capture.presence_mode(
                        stop_requested_getter=self.isInterruptionRequested
                    )
                    if is_present is not None:
                        self.person_presence_changed.emit(bool(is_present))
                    photo_capture.release_camera()

                    sleep_seconds = self._presence_check_interval_ms / 1000.0
                    slept = 0.0
                    while slept < sleep_seconds and not self.isInterruptionRequested():
                        if self._get_state() != WorkerState.PRESENCE_MONITORING:
                            break
                        time.sleep(0.1) # Intervall für die Anwesenheitsprüfung
                        slept += 0.1
                    continue
        except Exception as exc:
            print(f"YOLO Thread Error: {exc}")
        finally:
            photo_capture.release_camera()


class PipelineWorker(QThread):
    """
    Der PipelineWorker steuert den Datenfluss.
    Er fungiert als Überwachungsinstanz, die regelmäßig prüft, ob die
    lokalen KI-Dienste ihre Ergebnisse in die Inbox-Ordner geschrieben haben.
    """
    result_ready = pyqtSignal(str, list)
    reload_pool_requested = pyqtSignal()
    reload_pipeline_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.manager = None
        self._reload_lock = Lock()
        self._pool_reload_pending = False
        self._pipeline_reload_pending = False

    def request_pool_reload(self):
        with self._reload_lock:
            self._pool_reload_pending = True

    def request_pipeline_reload(self):
        with self._reload_lock:
            self._pipeline_reload_pending = True

    def _consume_reload_requests(self):
        with self._reload_lock:
            pool_reload = self._pool_reload_pending
            pipeline_reload = self._pipeline_reload_pending
            self._pool_reload_pending = False
            self._pipeline_reload_pending = False
        return pool_reload, pipeline_reload

    def run(self):
        from pipelinemanager import PipelineManager
        from pool_loader import PoolLoader

        try:
            with open("config.yaml", "r", encoding="utf-8") as handle:
                config_data = yaml.safe_load(handle)

            # PoolLoader lädt vordefinierte Textbausteine für die Kriminalgeschichten
            pool_loader = PoolLoader("config.yaml", config_data=config_data)
            # PipelineManager koordiniert die Logik der Dateiverarbeitung
            self.manager = PipelineManager(config_data, pool_loader=pool_loader)
            self.manager.data_finalized.connect(self.result_ready.emit)

            print("--- Pipeline: ACTIVE ---")
            while not self.isInterruptionRequested():
                pool_reload, pipeline_reload = self._consume_reload_requests()
                if self.manager is not None:
                    if pipeline_reload:
                        self.manager.reload_config()
                    if pool_reload:
                        self.manager.reload_pool_loader()
                # Suche nach neuen Dateien in den KI-Inboxes
                self.manager.check_for_updates()
                time.sleep(0.5)  # Kurze Pause zur Vermeidung von hoher CPU-Last
        except Exception as exc:
            print(f"Pipeline Thread Error: {exc}")


def run_app():
    """
    Hauptfunktion: Initialisiert die GUI, startet die Hintergrund-Worker
    und sorgt für die Vernetzung der Signale.
    """
    from gui.main_gui import ScalingAkteGUI
    from local_worker_manager import LocalWorkerManager

    yolo_thread = YOLOWorker()
    yolo_thread.prepare()

    app = QApplication(sys.argv)
    window = ScalingAkteGUI()
    window.show()
    app.processEvents()

    # Liste aller Verzeichnisse, die beim Start geleert werden müssen
    startup_cleanup_dirs = [
        "General ordner/final",
        "General ordner/main_image",
        "General ordner/sketch",
        "General ordner/ollama_ai/ollama_inbox",
        "General ordner/docker-compose-deepface/deepface_inbox",
        "General ordner/moondream_ai/moondream_inbox",
    ]
    for cleanup_dir in startup_cleanup_dirs:
        clear_directory(cleanup_dir)

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
    window.folder_closed.connect(yolo_thread.start_capture_mode)
    window.presence_monitoring_requested.connect(yolo_thread.start_presence_monitoring)

    time.sleep(0.5)
    yolo_thread.start()

    # Start der Qt-Eventschleife
    exit_code = app.exec()

    # Sauberes Herunterfahren der Threads bei Programmende
    pipeline_thread.requestInterruption()
    yolo_thread.requestInterruption()
    worker_manager.stop_workers()
    sys.exit(exit_code)


if __name__ == "__main__":
    run_app()
