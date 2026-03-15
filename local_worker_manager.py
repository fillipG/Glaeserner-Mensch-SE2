"""
LocalWorkerManager
------------------
Diese Klasse verwaltet die lokal laufenden Hilfsprozesse der Windows-Anwendung.
Im aktuellen Projekt betrifft das vor allem den lokalen Ollama-Dienst und den
zugehoerigen Python-Worker, der die dateibasierte Kommunikation mit der Pipeline
uebernimmt.

Zustaendigkeiten:
1. Pruefen, ob Ollama laut config.yaml aktiv genutzt werden soll.
2. Sicherstellen, dass der lokale Ollama-Dienst erreichbar ist.
3. Starten und Stoppen des lokalen workers/ollama_worker.py-Prozesses.
4. Uebernehmen von Config-Aenderungen ohne kompletten App-Neustart.

AUTOREN: Florian Hoeft
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import yaml


class LocalWorkerManager:
    """
    Kapselt alle lokal gestarteten Hilfsprozesse der Hauptanwendung.
    Dadurch muss main.py nicht selbst verwalten, ob der Ollama-Dienst schon
    laeuft, welches Modell konfiguriert ist und wann Worker sauber beendet
    werden muessen.
    """

    def __init__(self, repo_root=None, default_model="qwen2.5:3b"):
        self.repo_root = Path(repo_root or Path(__file__).resolve().parent)
        self.default_model = default_model
        self._ollama_worker_process = None
        self._ollama_service_process = None

    def start_ollama_worker(self):
        """
        Startet den lokalen Ollama-Worker fuer die dateibasierte Pipeline.
        Falls Ollama in der Config aktiv ist, wird vorher der lokale Dienst
        inklusive Modellverfuegbarkeit geprueft.
        """
        # Der lokale Worker laeuft immer, weil er auch den Pass-Through-Fall fuer deaktiviertes
        # Ollama uebernimmt. Der eigentliche Ollama-Preflight ist nur noetig, wenn das Modell
        # wirklich aktiv genutzt werden soll.
        if self._is_process_running(self._ollama_worker_process):
            return

        if self.is_ollama_enabled():
            model_name = self._load_configured_model()
            self._ensure_ollama_ready(model_name)

        script_path = self.repo_root / "workers" / "ollama_worker.py"
        if not script_path.exists():
            raise RuntimeError(f"Ollama-Worker nicht gefunden: {script_path}")

        self._ollama_worker_process = subprocess.Popen(
            [sys.executable, "-u", str(script_path)],
            cwd=str(self.repo_root),
            creationflags=self._windows_creation_flags(),
        )
        time.sleep(0.5)
        if self._ollama_worker_process.poll() is not None:
            raise RuntimeError(
                "Der lokale Ollama-Worker ist sofort beendet worden. "
                "Bitte pruefe die lokale Python-Umgebung fuer 'workers/ollama_worker.py'."
            )

    def stop_ollama_worker(self):
        """Beendet nur den lokalen Python-Worker, nicht aber zwingend den Ollama-Dienst."""
        self._stop_process(self._ollama_worker_process, "Ollama-Worker")
        self._ollama_worker_process = None

    def stop_workers(self):
        """Beendet alle durch den Manager gestarteten lokalen Hilfsprozesse."""
        self.stop_ollama_worker()
        self._stop_process(self._ollama_service_process, "Ollama-Dienst")
        self._ollama_service_process = None

    def sync_ollama_worker_state(self):
        """
        Uebernimmt Config-Aenderungen kontrolliert im laufenden Betrieb.
        Der Worker wird dazu einmal beendet und direkt mit der neuen Konfiguration
        wieder gestartet.
        """
        # Nach Config-Aenderungen wird der Worker einmal sauber neu gestartet, damit Toggle
        # und Modellwechsel ohne App-Neustart uebernommen werden.
        self.stop_ollama_worker()
        self.start_ollama_worker()

    def is_ollama_enabled(self):
        """Liest aus der zentralen Config, ob das Ollama-Modell aktiv sein soll."""
        config = self._load_config()
        pipeline = config.get("pipeline", [])
        if not isinstance(pipeline, list):
            return True
        for model_cfg in pipeline:
            if isinstance(model_cfg, dict) and model_cfg.get("id") == "ollama":
                return bool(model_cfg.get("enabled", True))
        return True

    def _ensure_ollama_ready(self, model_name):
        """
        Prueft die komplette lokale Ollama-Kette vor dem Worker-Start:
        Python-Modul, laufender API-Dienst und verfuegbares Modell.
        """
        # Vor echtem Ollama-Betrieb prueft der Manager bewusst die komplette Kette:
        # Python-Modul, lokaler API-Dienst und verfuegbares Modell.
        service_started = False
        if importlib.util.find_spec("ollama") is None:
            raise RuntimeError(
                "Das Python-Paket 'ollama' ist in dieser Umgebung nicht installiert. "
                "Bitte installiere es in der App-Umgebung."
            )
        if not self._is_ollama_api_ready():
            ollama_executable = shutil.which("ollama")
            if not ollama_executable:
                raise RuntimeError(
                    "Ollama ist nicht erreichbar und wurde nicht im PATH gefunden. "
                    "Bitte Ollama installieren oder manuell starten."
                )
            self._start_ollama_service(ollama_executable)
            service_started = True
            if not self._wait_for_ollama_api(timeout_seconds=15):
                raise RuntimeError(
                    "Ollama konnte nicht gestartet werden. "
                    "Bitte pruefe den lokalen Dienst auf http://localhost:11434."
                )

        available_models = self._load_available_models()
        if model_name not in available_models:
            raise RuntimeError(
                f"Das Modell '{model_name}' ist lokal nicht vorhanden. "
                f"Bitte fuehre 'ollama pull {model_name}' aus."
            )
        if service_started:
            print(f"[OLLAMA] Lokaler Dienst gestartet. API erreichbar, Modell: {model_name}")

    def _load_configured_model(self):
        """Liest das aktuell konfigurierte LLM-Modell mit Fallback auf das Standardmodell."""
        config = self._load_config()
        model_name = config.get("llm_model")
        if isinstance(model_name, str) and model_name.strip():
            return model_name.strip()
        return self.default_model

    def _load_config(self):
        """Liest config.yaml robust ein und faellt bei Fehlern auf ein leeres Dict zurueck."""
        config_path = self.repo_root / "config.yaml"
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        except Exception:
            return {}

    def _start_ollama_service(self, ollama_executable):
        """Startet bei Bedarf den lokalen Hintergrunddienst `ollama serve`."""
        if self._is_process_running(self._ollama_service_process):
            return

        print(f"[OLLAMA] Starte lokalen Dienst ueber: {ollama_executable} serve")
        self._ollama_service_process = subprocess.Popen(
            [ollama_executable, "serve"],
            cwd=str(self.repo_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=self._windows_creation_flags(),
        )

    def _wait_for_ollama_api(self, timeout_seconds):
        """Wartet mit Timeout darauf, dass die lokale Ollama-API erreichbar wird."""
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if self._is_ollama_api_ready():
                return True
            time.sleep(0.5)
        return False

    def _is_ollama_api_ready(self):
        """Prueft, ob der lokale Ollama-Dienst auf `127.0.0.1:11434` antwortet."""
        try:
            with urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as response:
                return response.status == 200
        except (URLError, OSError, ValueError):
            return False

    def _load_available_models(self):
        """Liest die lokal installierten Ollama-Modelle ueber die API aus."""
        try:
            with urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Ollama-Modelle konnten nicht abgefragt werden: {exc}") from exc

        models = payload.get("models", [])
        names = set()
        for model in models:
            if not isinstance(model, dict):
                continue
            name = model.get("name")
            if isinstance(name, str) and name.strip():
                names.add(name.strip())
        return names

    def _stop_process(self, process, label):
        """Beendet einen laufenden Unterprozess kontrolliert mit Terminate/Kill-Fallback."""
        if not self._is_process_running(process):
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        print(f"{label} beendet.")

    @staticmethod
    def _is_process_running(process):
        """Hilfsfunktion fuer den einheitlichen Running-Check von Unterprozessen."""
        return process is not None and process.poll() is None

    @staticmethod
    def _windows_creation_flags():
        """Verhindert auf Windows zusaetzliche Konsolenfenster fuer Hintergrundprozesse."""
        if os.name != "nt":
            return 0
        return subprocess.CREATE_NO_WINDOW
