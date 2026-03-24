"""
workers/pipeline_worker.py
---------------------------
Datenfluss-Überwachungs-Thread der Anwendung.

Der PipelineWorker läuft im Hintergrund und überwacht die KI-Ergebnis-Ordner
auf neue Dateien. Er koordiniert den Datenfluss zwischen den KI-Modulen
(Deepface, Moondream, Ollama) und der GUI.

Ablauf:
    1. PipelineManager wird gestartet
    2. PipelineManager.check_for_updates() wird alle 0.5 Sekunden aufgerufen
    3. Wenn alle KI-Ergebnisse für eine Person vorliegen, wird data_finalized.emit() ausgelöst
    4. GUI empfängt das Signal und zeigt die Ergebnisse an

Reload-Mechanismus:
    Konfigurations- und Pool-Änderungen aus dem Admin-Menü werden über
    request_pool_reload() / request_pipeline_reload() weitergeleitet.
    Dies geschieht thread-safe über Flags, die in der run()-Schleife abgefragt werden.

Signale:
    result_ready(str, list) → Status + Personendaten an die GUI

AUTOREN: Dennis Penner, Florian Hoeft
"""

import time
import yaml
from threading import Lock

from PyQt6.QtCore import QThread, pyqtSignal


class PipelineWorker(QThread):
    """
    Überwacht die KI-Inbox-Ordner und koordiniert die Datenzusammenführung.
    Sendet fertige Ergebnisse über das result_ready-Signal an die GUI.
    """
    result_ready = pyqtSignal(str, list)

    def __init__(self):
        super().__init__()
        self.manager = None
        self._reload_lock = Lock()
        self._pool_reload_pending = False
        self._pipeline_reload_pending = False

    def request_pool_reload(self):
        """
        Thread-safe: markiert, dass der Pool in der nächsten Runde neu geladen werden soll.
        Wird vom GUI-Thread aufgerufen, wenn Pool-Einstellungen geändert wurden.
        """
        with self._reload_lock:
            self._pool_reload_pending = True

    def request_pipeline_reload(self):
        """
        Thread-safe: markiert, dass die Pipeline-Config in der nächsten Runde neu geladen werden soll.
        Wird vom GUI-Thread aufgerufen, wenn Pipeline-Modelle aktiviert/deaktiviert wurden.
        """
        with self._reload_lock:
            self._pipeline_reload_pending = True

    def _consume_reload_requests(self):
        """Liest und leert beide Reload-Flags atomar."""
        with self._reload_lock:
            pool_reload = self._pool_reload_pending
            pipeline_reload = self._pipeline_reload_pending
            self._pool_reload_pending = False
            self._pipeline_reload_pending = False
        return pool_reload, pipeline_reload

    def run(self):
        """
        Hauptschleife: initialisiert PipelineManager und fragt alle 0.5s nach neuen Dateien.
        Der 0.5s-Takt ist ein bewusster Kompromiss: schnell genug für flüssigen Betrieb,
        langsam genug um die CPU im Dauerbetrieb nicht zu überlasten.
        """
        from pipelinemanager import PipelineManager
        from pool_loader import PoolLoader

        try:
            with open("config.yaml", "r", encoding="utf-8") as handle:
                config_data = yaml.safe_load(handle)

            # PoolLoader verwaltet die vordefinierten Fallback-Personen (aus ./pool/)
            pool_loader = PoolLoader("config.yaml", config_data=config_data)
            # PipelineManager koordiniert die Dateiverarbeitung und den Batch-Abschluss
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
                    # Suche nach neuen Dateien in den KI-Inbox-Ordnern
                    self.manager.check_for_updates()
                # 0.5s Pause verhindert busy-wait und schont die CPU im 24/7-Betrieb
                time.sleep(0.5)

        except Exception as exc:
            print(f"Pipeline Thread Error: {exc}")
