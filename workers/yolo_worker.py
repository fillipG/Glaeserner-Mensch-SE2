"""
workers/yolo_worker.py
----------------------
Kamera-Thread der Anwendung.

Der YOLOWorker kapselt die gesamte Bildverarbeitung der Kamera.
Er läuft in einem eigenen QThread, damit die GUI während der
Personenerkennung und dem Foto-Countdown nicht einfriert.

State Machine:
    IDLE               → Kamera aktiv, sucht Personen. Bei Erkennung: Countdown + Foto.
    ANALYZING          → Foto gemacht, KIs rechnen. Kamera wird geschont.
    PRESENCE_MONITORING→ Ergebnisse werden angezeigt, prüft ob Person noch da ist.

Signale:
    frame_ready(frame)          → Live-Bild für Camera-Preview in der GUI
    photo_done()                → Foto erfolgreich gespeichert
    person_presence_changed(bool) → True = Person da, False = Person weg
    startup_error(str)          → Fehlermeldung bei Hardware-Initialisierung

AUTOREN: Dennis Penner, Florian Hoeft
"""

import os
import time
from enum import Enum

import yaml
from PyQt6.QtCore import QThread, pyqtSignal


class WorkerState(Enum):
    """
    Zustände des Kamera-Workers.
    Die State Machine verhindert Race Conditions zwischen Kamera und GUI.
    """
    IDLE                = "idle"                 # Wartet auf Person
    ANALYZING           = "analyzing"            # Foto gemacht, KIs arbeiten
    PRESENCE_MONITORING = "presence_monitoring"  # Prüft, ob Nutzer noch vor Ort ist


class YOLOWorker(QThread):
    """
    Kamera- und Personen-Erkennungs-Thread.

    Dieser Thread läuft dauerhaft im Hintergrund. Er wechselt zwischen
    drei Zuständen (WorkerState) und reagiert damit auf die aktuellen
    Phasen der Museumsanwendung.
    """
    frame_ready               = pyqtSignal(object)  # Live-Bild → GUI Camera-Preview
    photo_done                = pyqtSignal()         # Foto erfolgreich → GUI startet Analyse-Anzeige
    person_presence_changed   = pyqtSignal(bool)     # True = Person da, False = Person weg
    startup_error             = pyqtSignal(str)      # Fehler bei YOLO/Kamera-Initialisierung

    def __init__(self):
        super().__init__()
        self._state = WorkerState.IDLE
        self._presence_check_interval_ms = 2000
        self._photo_capture = None
        self._startup_error_message = None
        self._camera_prewarm_requested = False
        self._keep_camera_open_until_idle = False

    def _get_state(self):
        return self._state

    def _set_state(self, state):
        self._state = state

    def start_capture_mode(self):
        """Versetzt den Worker zurück in den Suchmodus (z.B. nach einem Reset der GUI)."""
        print("YOLO Worker: IDLE")
        self._camera_prewarm_requested = False
        self._keep_camera_open_until_idle = False
        self._set_state(WorkerState.IDLE)

    def start_analyzing_mode(self):
        """
        Wechselt in den Wartezustand während die KIs arbeiten.
        Kamera wird dabei meist freigegeben, kann bei aktivem Prewarm
        aber bereits wieder geöffnet werden.
        """
        print("YOLO Worker: ANALYZING")
        self._keep_camera_open_until_idle = False
        self._set_state(WorkerState.ANALYZING)

    def start_presence_monitoring(self):
        """
        Aktiviert die Anwesenheits-Prüfung nach Abschluss der Analyse.
        Prüft in konfigurierbaren Abständen, ob die Person noch vor dem Gerät steht.
        """
        print("YOLO Worker: PRESENCE_MONITORING")
        self._set_state(WorkerState.PRESENCE_MONITORING)

    def request_camera_prewarm(self):
        """
        Fordert ein Vorwaermen der Kamera an.

        Die eigentliche Initialisierung bleibt im Worker-Thread, damit
        die GUI beim Schliessen der Mappe nicht blockiert.
        """
        self._camera_prewarm_requested = True

    def _prewarm_camera_if_requested(self, photo_capture):
        """
        Oeffnet die Kamera im Hintergrund und liest mehrere Test-Frames an.

        So kann die Treiber- und Aufloesungsinitialisierung bereits waehrend
        der Schliess-Animation passieren, ohne Frames an die GUI zu senden.
        """
        if not self._camera_prewarm_requested or photo_capture is None:
            return

        cap = photo_capture.ensure_camera_open(log_open=False)
        if not cap:
            return

        # Viele Kameratreiber liefern die ersten ein bis zwei Frames noch nicht
        # stabil. Liest deshalb bis zu drei Frames an, damit das erste echte
        # Livebild nach dem Schliessen moeglichst sofort sichtbar wird.
        for _ in range(3):
            ret, _ = cap.read()
            if not ret:
                continue
            self._camera_prewarm_requested = False
            self._keep_camera_open_until_idle = True
            return

        photo_capture.release_camera()

    def prepare(self):
        """
        Lädt das YOLO-Modell vorab (im Hauptthread, vor dem Thread-Start).
        Wird in run_app() aufgerufen, damit Ladefehler sofort sichtbar sind.
        """
        if self._photo_capture is not None or self._startup_error_message is not None:
            return

        try:
            from ultralytics import YOLO
            from PersonPhotoCapture import PersonPhotoCapture

            from path_service import get_paths
            os.makedirs(get_paths()["main_image"], exist_ok=True)
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
        Liest bei jedem Durchlauf die config.yaml neu ein, damit Änderungen
        im Admin-Menü ohne Neustart der Anwendung wirksam werden.
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
                # Config wird jede Runde frisch gelesen, damit Laufzeit-Änderungen
                # aus dem Admin-Menü sofort greifen (z.B. Countdown-Dauer, Sprache).
                with open("config.yaml", "r", encoding="utf-8") as handle:
                    cfg = yaml.safe_load(handle) or {}

                photo_delay = int(cfg.get("photo_delay") or 3)
                language = cfg.get("language", "de")
                self._presence_check_interval_ms = max(
                    1000,
                    int(cfg.get("no_person_check_interval_ms", 2000) or 2000),
                )
                photo_capture.update_runtime_config(photo_delay=photo_delay, language=language)

                state = self._get_state()

                # Start-Prewarm und vorgewaermte Rueckkehr nach dem Schliessen greifen
                # auch im IDLE-Zustand, damit capture_mode eine bereits offene Kamera nutzt.
                if state == WorkerState.IDLE:
                    self._prewarm_camera_if_requested(photo_capture)

                # FALL 1: Suche nach Personen & Automatischer Snapshot
                if state == WorkerState.IDLE:
                    captured_frame = photo_capture.capture_mode(
                        frame_callback=lambda frame: self.frame_ready.emit(frame),
                        stop_requested_getter=self.isInterruptionRequested,
                        mode_active_getter=lambda: self._get_state() == WorkerState.IDLE,
                    )
                    if captured_frame is None:
                        # Kein Frame → kurze Pause bevor nächster Versuch
                        time.sleep(0.05)
                        continue

                    import cv2
                    # Bild zentral speichern, damit KI-Docker-Container darauf zugreifen können
                    from path_service import get_paths
                    filename = str(get_paths()["main_image"] / "face_trigger.jpg")
                    cv2.imwrite(filename, captured_frame)
                    print(f"[{time.strftime('%H:%M:%S')}] Bild gespeichert: {filename}")
                    photo_capture.release_camera()
                    self.start_analyzing_mode()
                    self.photo_done.emit()
                    continue

                # FALL 2: Wartezustand während der KI-Analyse (Ressourcenschonung)
                if state == WorkerState.ANALYZING:
                    self._prewarm_camera_if_requested(photo_capture)
                    if not self._keep_camera_open_until_idle:
                        photo_capture.release_camera()
                    time.sleep(0.1)
                    continue

                # FALL 3: Anwesenheits-Prüfung (ist die Person noch da?)
                if state == WorkerState.PRESENCE_MONITORING:
                    # Beim Schliessen der Mappe ist schnelles Vorwaermen wichtiger
                    # als weitere Presence-Checks, weil als naechstes ohnehin IDLE folgt.
                    if self._camera_prewarm_requested or self._keep_camera_open_until_idle:
                        self._prewarm_camera_if_requested(photo_capture)
                        time.sleep(0.05)
                        continue

                    is_present = photo_capture.presence_mode(
                        stop_requested_getter=self.isInterruptionRequested
                    )
                    if is_present is not None:
                        self.person_presence_changed.emit(bool(is_present))
                    self._prewarm_camera_if_requested(photo_capture)
                    if not self._keep_camera_open_until_idle:
                        photo_capture.release_camera()

                    # Wartezeit zwischen zwei Anwesenheitsprüfungen (aus Config steuerbar)
                    sleep_seconds = self._presence_check_interval_ms / 1000.0
                    slept = 0.0
                    while slept < sleep_seconds and not self.isInterruptionRequested():
                        if self._camera_prewarm_requested:
                            break
                        if self._get_state() != WorkerState.PRESENCE_MONITORING:
                            break
                        time.sleep(0.1)
                        slept += 0.1
                    continue

        except Exception as exc:
            print(f"YOLO Thread Error: {exc}")
        finally:
            if photo_capture is not None:
                photo_capture.release_camera()
