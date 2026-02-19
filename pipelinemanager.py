import os
import time
import yaml
from service import TranslationService

class PipelineManager:
    """
    Diese Klasse fungiert als zentrale Steuerungseinheit.
    Sie überwacht die Ausgaben verschiedener KI-Modelle, führt diese zusammen
    und übernimmt optional die Lokalisierung (Übersetzung) der Ergebnisse.
    """
    def __init__(self, config_data):
        # Initialisierung der Übersetzungslogik basierend auf der Konfiguration
        self.target_lang = config_data.get("language", "en")
        if self.target_lang == "de":
            # Instanziierung des externen Service-Objekts für die Übersetzung
            self.translator = TranslationService(target_lang='de')
        else:
            self.translator = None

        # Identifikation der aktiven Pipeline-Komponenten aus der config.yaml
        self.enabled_models = [cfg for cfg in config_data["pipeline"] if cfg.get("enabled", False)]
        self.required_ids = [m["id"] for m in self.enabled_models]

        # Konfiguration des Überwachungsverzeichnisses
        self.watch_dir = os.path.abspath(self.enabled_models[0]["watch_dir"])
        self.file_ext = ".yaml"

        # Zwischenspeicher zur Gruppierung der Modellergebnisse pro Face-ID
        self.results_cache = {}

        # Initialer Scan: Bestehende Dateien werden ignoriert
        self.seen_files = {f for f in os.listdir(self.watch_dir) if f.endswith(self.file_ext)}
        os.makedirs(self.watch_dir, exist_ok=True)

        print(f"PIPELINE INITIALIZED")
        print(f"Waiting for models: {', '.join(self.required_ids).upper()}")
        print(f"Watching folder: {self.watch_dir}\n")

    def check_for_updates(self):
        """Überprüft das Zielverzeichnis auf neue Dateien."""
        try:
            current_files = {f for f in os.listdir(self.watch_dir) if f.endswith(self.file_ext)}
            new_files = current_files - self.seen_files

            for file_name in new_files:
                time.sleep(0.1)  # Buffer für Schreibvorgang
                self.process_incoming_file(file_name)
                self.seen_files.add(file_name)
        except Exception as e:
            print(f"Error scanning: {e}")

    def process_incoming_file(self, file_name):
        """Analysiert den Dateinamen zur Zuordnung der Metadaten."""
        try:
            name_no_ext = file_name.replace(self.file_ext, "")
            if "_" not in name_no_ext: return

            base_id, model_id = name_no_ext.split("_", 1)

            if base_id not in self.results_cache:
                self.results_cache[base_id] = {}

            file_path = os.path.join(self.watch_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                self.results_cache[base_id][model_id] = data.get("description") if isinstance(data, dict) else data

            received = list(self.results_cache[base_id].keys())
            waiting_for = [m for m in self.required_ids if m not in received]

            print(f"[{base_id.upper()}] Received: {model_id.upper()}")
            if not waiting_for:
                self.finalize_group(base_id)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    def finalize_group(self, base_id):
        """Aggregation, Lokalisierung und finale Konsolenausgabe."""
        print(f"\n[COMPLETE ANALYSIS] {base_id.upper()}")
        print("=" * 60)
        for m_id in self.required_ids:
            content = self.results_cache[base_id][m_id]
            if self.translator and self.target_lang == "de":
                content = self.translator.translate_text(content)
            print(f"{m_id.upper()}: {content}")
        print("=" * 60 + "\n")
        del self.results_cache[base_id]