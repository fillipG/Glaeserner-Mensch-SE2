import os
import time
import yaml
from PyQt6.QtCore import QObject, pyqtSignal

from service import TranslationService

"""
PIPELINEMANAGER-KLASSE:
Dieser Worker läuft in einem eigenen Hintergrund-Thread (QThread).
Seine Aufgabe ist es, kontinuierlich den "final"-Ordner zu überwachen,
KI-Ergebnisse (YAML-Dateien) zusammenzuführen und die fertigen
Daten an die GUI zu senden.
"""
class PipelineManager(QObject):

    # Signal: Sendet Face-ID (z.B. 'face1') und Ergebnis an die GUI
    data_finalized = pyqtSignal(str, dict)

    def __init__(self, config_data):
        super().__init__()

        self.target_lang = config_data.get("language", "en")
        if self.target_lang == "de":
            self.translator = TranslationService(target_lang='de')
        else:
            self.translator = None

        # Aktive Modelle aus Config laden
        self.enabled_models = [cfg for cfg in config_data["pipeline"] if cfg.get("enabled", False)]
        self.required_ids = [m["id"] for m in self.enabled_models]

        # Pfad-Konfiguration
        self.watch_dir = os.path.abspath(self.enabled_models[0]["watch_dir"])
        self.file_ext = ".yaml"

        # Zwischenspeicher für unvollständige Modell-Gruppen
        self.results_cache = {}

        # Wir starten mit einem leeren Set
        # Dadurch werden beim ersten Scan auch bereits existierende Dateien verarbeitet
        self.seen_files = set()

        os.makedirs(self.watch_dir, exist_ok=True)

        print(f"PIPELINE INITIALIZED")
        print(f"Waiting for models: {', '.join(self.required_ids).upper()}")
        print(f"Watching folder: {self.watch_dir}\n")

    def check_for_updates(self):
        """Überprüft den Ordner auf neue, geänderte oder gelöschte Dateien"""
        try:
            # Aktuellen Ordnerinhalt lesen
            current_files = {f for f in os.listdir(self.watch_dir) if f.endswith(self.file_ext)}

            # Identifiziere neue Dateien
            new_files = current_files - self.seen_files

            # Identifiziere gelöschte Dateien
            removed_files = self.seen_files - current_files
            for f in removed_files:
                self.seen_files.remove(f)

            # Verarbeite alle neuen Dateien
            for file_name in new_files:
                time.sleep(0.05)  # Buffer für Dateisystem-Schreibvorgänge
                self.process_incoming_file(file_name)
                self.seen_files.add(file_name)

        except Exception as e:
            print(f"Error scanning: {e}")

    def process_incoming_file(self, file_name):
        """Liest eine einzelne YAML Datei aus und ordnet sie einer Gruppe zu"""
        try:
            name_no_ext = file_name.replace(self.file_ext, "")
            if "_" not in name_no_ext: return

            # Trennung von faceID und Modell
            base_id, model_id = name_no_ext.split("_", 1)

            if base_id not in self.results_cache:
                self.results_cache[base_id] = {}

            file_path = os.path.join(self.watch_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

                if isinstance(data, dict):
                    self.results_cache[base_id][model_id] = data
                else:
                    # Fallback für einfache Text-Dateien
                    self.results_cache[base_id][model_id] = {"description": str(data), "prompt": "N/A"}

            # Prüfen, ob alle erwarteten Modelle für diese ID geliefert haben
            received = list(self.results_cache[base_id].keys())
            waiting_for = [m for m in self.required_ids if m not in received]

            print(f"[{base_id.upper()}] Received: {model_id.upper()}")

            if not waiting_for:
                self.finalize_group(base_id)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    def finalize_group(self, base_id):
        """Führt alle Modellergebnisse zusammen, übersetzt sie ggf. und sendet sie an die GUI."""
        final_results = {}

        print(f"\n[START ANALYSIS] ID: {base_id.upper()}")
        print("=" * 60)

        captured_data = self.results_cache.get(base_id, {})

        for m_id in self.required_ids:
            model_data = captured_data.get(m_id, {})

            # Dynamische Feld-Extraktion
            if m_id == "deepface":
                emotion = model_data.get("Emotion", "N/A")
                alter = model_data.get("Alter", "N/A")
                geschlecht = model_data.get("Geschlecht", "N/A")
                content = f"Emotion: {emotion}, Alter: {alter}, Geschlecht: {geschlecht}"
            else:
                content = model_data.get("description", "No description found")

            # Lokalisierung (Übersetzung)
            if self.translator and self.target_lang == "de":
                # Deepface Felder sind bereits Deutsch, daher nur Moondream übersetzen
                if m_id != "deepface":
                    content = self.translator.translate_text(content)

            final_results[m_id] = content

            # Konsolenausgabe für das einzelne Modell
            print(f"  > MODEL:  {m_id.upper()}")
            print(f"    RESULT: {content}")
            print("-" * 30)

        # --- ABSCHLUSS-MELDUNG ---
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        print(f"DONE: Analysis for {base_id.upper()} completed at {timestamp}.")
        print("=" * 60 + "\n")

        # Daten per Signal an die GUI senden
        self.data_finalized.emit(base_id, final_results)

        # Cache leeren
        del self.results_cache[base_id]