import os
import sys
import time
from enum import Enum

import cv2
import yaml
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMessageBox


def clear_directory(path):
    """Loescht den Inhalt eines Ordners, laesst den Ordner selbst aber bestehen."""
    os.makedirs(path, exist_ok=True)
    for entry in os.listdir(path):
        entry_path = os.path.join(path, entry)
        try:
            if os.path.isfile(entry_path) or os.path.islink(entry_path):
                os.unlink(entry_path)
            elif os.path.isdir(entry_path):
                for root, dirs, files in os.walk(entry_path, topdown=False):
                    for file_name in files:
                        os.unlink(os.path.join(root, file_name))
                    for dir_name in dirs:
                        os.rmdir(os.path.join(root, dir_name))
                os.rmdir(entry_path)
        except Exception as exc:
            print(f"Startup-Cleanup konnte {entry_path} nicht loeschen: {exc}")


class WorkerState(Enum):
    IDLE = "idle"
    ANALYZING = "analyzing"
    PRESENCE_MONITORING = "presence_monitoring"


class YOLOWorker(QThread):
    frame_ready = pyqtSignal(object)
    photo_done = pyqtSignal()
    person_presence_changed = pyqtSignal(bool)
    startup_error = pyqtSignal(str)

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
        print("YOLO Worker: IDLE")
        self._set_state(WorkerState.IDLE)

    def start_analyzing_mode(self):
        print("YOLO Worker: ANALYZING")
        self._set_state(WorkerState.ANALYZING)

    def start_presence_monitoring(self):
        print("YOLO Worker: PRESENCE_MONITORING")
        self._set_state(WorkerState.PRESENCE_MONITORING)

    def prepare(self):
        if self._photo_capture is not None or self._startup_error_message is not None:
            return

        try:
            from PersonPhotoCapture import PersonPhotoCapture

            os.makedirs("General ordner/main_image", exist_ok=True)
            print("YOLO Worker: bereite Modell im Hauptthread vor...")
            self._photo_capture = PersonPhotoCapture(photo_delay=3)
        except Exception as exc:
            self._startup_error_message = (
                "YOLO/Torch konnte nicht initialisiert werden.\n"
                "Wahrscheinlich fehlt auf diesem Windows-System eine Torch-Abhaengigkeit "
                "oder es ist eine unpassende Torch-Installation aktiv.\n"
                f"Details: {exc}"
            )

    def run(self):
        self.prepare()
        if self._startup_error_message is not None:
            print(self._startup_error_message)
            self.startup_error.emit(self._startup_error_message)
            return

        photo_capture = self._photo_capture
        print("--- YOLO Worker: ACTIVE ---")

        try:
            while not self.isInterruptionRequested():
                with open("config.yaml", "r", encoding="utf-8") as handle:
                    cfg = yaml.safe_load(handle) or {}

                photo_delay = int(cfg.get("photo_delay") or 3)
                self._presence_check_interval_ms = max(
                    1000,
                    int(cfg.get("no_person_check_interval_ms", 2000) or 2000),
                )
                photo_capture.update_runtime_config(photo_delay=photo_delay)

                state = self._get_state()
                if state == WorkerState.IDLE:
                    captured_frame = photo_capture.capture_mode(
                        frame_callback=lambda frame: self.frame_ready.emit(frame),
                        stop_requested_getter=self.isInterruptionRequested,
                        mode_active_getter=lambda: self._get_state() == WorkerState.IDLE,
                    )
                    if captured_frame is None:
                        time.sleep(0.05)
                        continue

                    filename = "General ordner/main_image/face_trigger.jpg"
                    cv2.imwrite(filename, captured_frame)
                    print(f"Bild gespeichert: {filename}")
                    photo_capture.release_camera()
                    self.start_analyzing_mode()
                    self.photo_done.emit()
                    continue

                if state == WorkerState.ANALYZING:
                    photo_capture.release_camera()
                    time.sleep(0.1)
                    continue

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
                        time.sleep(0.1)
                        slept += 0.1
                    continue
        except Exception as exc:
            print(f"YOLO Thread Error: {exc}")
        finally:
            photo_capture.release_camera()


class PipelineWorker(QThread):
    result_ready = pyqtSignal(str, list)
    reload_pool_requested = pyqtSignal()
    reload_pipeline_requested = pyqtSignal()

    def request_pool_reload(self):
        self.reload_pool_requested.emit()

    def request_pipeline_reload(self):
        self.reload_pipeline_requested.emit()

    def run(self):
        from pipelinemanager import PipelineManager
        from pool_loader import PoolLoader

        try:
            with open("config.yaml", "r", encoding="utf-8") as handle:
                config_data = yaml.safe_load(handle)

            pool_loader = PoolLoader("config.yaml", config_data=config_data)
            self.manager = PipelineManager(config_data, pool_loader=pool_loader)
            self.manager.data_finalized.connect(self.result_ready.emit)
            self.reload_pool_requested.connect(
                self.manager.reload_pool_loader,
                Qt.ConnectionType.QueuedConnection,
            )
            self.reload_pipeline_requested.connect(
                self.manager.reload_config,
                Qt.ConnectionType.QueuedConnection,
            )

            print("--- Pipeline Worker: ACTIVE ---")
            while not self.isInterruptionRequested():
                self.manager.check_for_updates()
                time.sleep(0.5)
        except Exception as exc:
            print(f"Pipeline Thread Error: {exc}")


def run_app():
    from gui.main_gui import ScalingAkteGUI
    from local_worker_manager import LocalWorkerManager

    yolo_thread = YOLOWorker()
    yolo_thread.prepare()

    app = QApplication(sys.argv)
    window = ScalingAkteGUI()
    window.show()
    app.processEvents()

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
    app.aboutToQuit.connect(worker_manager.stop_workers)

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

    exit_code = app.exec()
    pipeline_thread.requestInterruption()
    yolo_thread.requestInterruption()
    worker_manager.stop_workers()
    sys.exit(exit_code)


if __name__ == "__main__":
    run_app()
