import os
import time
import yaml
import shutil
from PyQt6.QtCore import QObject, pyqtSignal

from service import TranslationService

"""
PIPELINEMANAGER-KLASSE:
Dieser Worker läuft in einem eigenen Hintergrund-Thread (QThread).
Seine Aufgabe ist es, kontinuierlich den "final"-Ordner zu überwachen,
KI-Ergebnisse (YAML-Dateien) zusammenzuführen, die fertigen
Daten an die GUI zu senden und danach die Ergebnisse zu löschen.
"""
class PipelineManager(QObject):
    # Signal: Sendet einen Status-String und die komplette Personen-Liste
    data_finalized = pyqtSignal(str, list)

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

        # Speicher für die Verarbeitung
        self.results_cache = {}
        self.collected_faces = []
        self.expected_face_count = 0
        self.seen_files = set()

        os.makedirs(self.watch_dir, exist_ok=True)

        print(f"PIPELINE INITIALIZED")
        print(f"Waiting for models: {', '.join(self.required_ids).upper()}")
        print(f"Watching folder: {self.watch_dir}\n")

    def check_for_updates(self):
        """Überprüft den Ordner auf neue Dateien."""
        try:
            # 1. Prüfen, ob das Log existiert, um die Zielanzahl zu kennen
            log_path = os.path.join(self.watch_dir, "faces_log.yaml")
            if os.path.exists(log_path) and log_path not in self.seen_files:
                with open(log_path, 'r') as f:
                    log_data = yaml.safe_load(f)
                    self.expected_face_count = log_data.get("face_count", 0)
                    print(f"[LOG] Expecting {self.expected_face_count} faces in total.")
                self.seen_files.add(log_path)

            # 2. Dateien scannen
            current_files = {f for f in os.listdir(self.watch_dir) if f.endswith(self.file_ext)}
            new_files = current_files - self.seen_files

            for file_name in new_files:
                if file_name == "faces_log.yaml": continue
                time.sleep(0.05)
                self.process_incoming_file(file_name)
                self.seen_files.add(file_name)

        except Exception as e:
            print(f"Error scanning: {e}")

    def process_incoming_file(self, file_name):
        """Liest YAML und prüft, ob ein Gesicht vollständig analysiert ist."""
        try:
            name_no_ext = file_name.replace(self.file_ext, "")
            if "_" not in name_no_ext: return

            base_id, model_id = name_no_ext.split("_", 1)

            if base_id not in self.results_cache:
                self.results_cache[base_id] = {}

            file_path = os.path.join(self.watch_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                self.results_cache[base_id][model_id] = data if isinstance(data, dict) else {"description": str(data)}

            # Prüfen, ob alle Modelle für diese ID geliefert haben
            received = list(self.results_cache[base_id].keys())
            waiting_for = [m for m in self.required_ids if m not in received]

            print(f"[{base_id.upper()}] Received: {model_id.upper()}")

            if not waiting_for:
                self.add_to_batch(base_id)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    def add_to_batch(self, base_id):
        """Bereitet ein Gesicht vor und fügt es der Sammel-Liste hinzu."""
        captured_data = self.results_cache.get(base_id, {})
        df_data = captured_data.get("deepface", {})
        moon_data = captured_data.get("moondream", {})

        person_dict = {
            "titel": f"ID: {base_id.upper()}",
            "geschlecht": df_data.get("Geschlecht", "Unbekannt"),
            "augen": "Braun",
            "stimmung": df_data.get("Emotion", "Neutral"),
            "alter": str(df_data.get("Alter", "N/A")),
            "gefahr": self._calculate_danger(df_data.get("Emotion", "Neutral")),
            "beschreibung": moon_data.get("description", "Keine Beschreibung gefunden.")
        }

        if self.translator and self.target_lang == "de":
            person_dict["beschreibung"] = self.translator.translate_text(person_dict["beschreibung"])

        self.collected_faces.append(person_dict)
        print(f"--- [COLLECTED] {base_id} ({len(self.collected_faces)}/{self.expected_face_count}) ---")

        if self.expected_face_count > 0 and len(self.collected_faces) >= self.expected_face_count:
            self.finalize_and_send_batch()

    def _calculate_danger(self, emotion):
        danger_map = {
            "Wütend": "HOCH",
            "Beunruhigt": "MITTEL",
            "Angst": "MITTEL",
            "Ekel": "GERING"
        }
        return danger_map.get(emotion, "GERING")

    def finalize_and_send_batch(self):
        if not self.collected_faces:
            return

        print(f"\n🚀 ALL FACES READY! Sending batch of {len(self.collected_faces)} to GUI...")
        data_to_send = list(self.collected_faces)
        self.data_finalized.emit("BATCH", data_to_send)

        # Pause, damit die GUI die Daten laden kann, bevor wir die Dateien löschen
        time.sleep(2.0)
        self.cleanup_folders()

    def cleanup_folders(self):
        """Löscht nur den final-Ordner und setzt den internen Status zurück."""
        final_folder = os.path.abspath(self.watch_dir)

        print(f"\n🧹 Cleaning up: {os.path.basename(final_folder)}")

        if os.path.exists(final_folder):
            for filename in os.listdir(final_folder):
                file_path = os.path.join(final_folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"      [SKIP] {filename} is busy: {e}")

        # Wichtig: Internen Speicher zurücksetzen
        self.seen_files.clear()
        self.results_cache.clear()
        self.collected_faces = []
        self.expected_face_count = 0

        print("✨ System reset and ready for next person.")