import os
import time
import yaml
import shutil
from PyQt6.QtCore import QObject, pyqtSignal

from service import TranslationService

"""
PIPELINEMANAGER-KLASSE:
Überwacht den final-Ordner, sammelt KI-Ergebnisse von allen aktiven Modellen,
sendet die fertigen Datensätze an die GUI und löscht die Dateien danach.
"""
class PipelineManager(QObject):
    # Signal: Sendet Status-String und komplette Personenliste an GUI
    data_finalized = pyqtSignal(str, list)

    def __init__(self, config_data, pool_loader=None):
        super().__init__()
        self.pool_loader = pool_loader

        self.target_lang = config_data.get("language", "en")
        if self.target_lang == "de":
            self.translator = TranslationService(target_lang='de')
        else:
            self.translator = None

        # Aktive Modelle aus Config
        self.enabled_models = [cfg for cfg in config_data["pipeline"] if cfg.get("enabled", False)]
        self.required_ids = [m["id"] for m in self.enabled_models]

        # Ordner, der auf neue YAML-Dateien überwacht wird
        self.watch_dir = os.path.abspath(self.enabled_models[0]["watch_dir"])
        self.file_ext = ".yaml"

        # Interner Speicher für Ergebnisse
        self.results_cache = {}
        self.collected_faces = []
        self.expected_face_count = 0
        self.seen_files = set()

        os.makedirs(self.watch_dir, exist_ok=True)

        print(f"PIPELINE INITIALIZED")
        print(f"Waiting for models: {', '.join(self.required_ids).upper()}")
        print(f"Watching folder: {self.watch_dir}\n")

    # =========================================================
    # THREAD LOOP: Prüft regelmäßig auf neue Dateien
    # =========================================================
    def check_for_updates(self):
        try:
            log_name = "faces_log.yaml"
            log_path = os.path.join(self.watch_dir, log_name)

            # Log-Datei prüfen, um Anzahl der erwarteten Gesichter zu lesen
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    log_data = yaml.safe_load(f) or {}

                new_face_count = log_data.get("face_count", 0)
                # Reset results only if we detect a new batch
                if new_face_count != self.expected_face_count:
                    self.results_cache.clear()
                    self.collected_faces.clear()
                    self.seen_files.clear()

                self.expected_face_count = new_face_count
                print(f"[LOG] Expecting {self.expected_face_count} faces in total.")

            # Alle neuen YAML-Dateien scannen
            current_files = {f for f in os.listdir(self.watch_dir) if f.endswith(self.file_ext)}
            new_files = current_files - self.seen_files

            for file_name in new_files:
                if file_name == log_name:
                    continue
                time.sleep(0.05)
                self.process_incoming_file(file_name)
                self.seen_files.add(file_name)

        except Exception as e:
            print(f"Error scanning: {e}")

    # =========================================================
    # Einzelne Datei verarbeiten und prüfen, ob alle Modelle geliefert haben
    # =========================================================
    def process_incoming_file(self, file_name):
        try:
            name_no_ext = file_name.replace(self.file_ext, "")
            if "_" not in name_no_ext:
                return

            base_id, model_id = name_no_ext.rsplit("_", 1)

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

    # =========================================================
    # Gesicht fertig vorbereiten und der Sammelliste hinzufügen
    # =========================================================
    def add_to_batch(self, base_id):
        captured_data = self.results_cache.get(base_id, {})
        df_data = captured_data.get("deepface", {})
        moon_data = captured_data.get("moondream", {})

        person_dict = self._build_person_dict(base_id, df_data, moon_data)
        self.collected_faces.append(person_dict)
        print(f"--- [COLLECTED] {base_id} ({len(self.collected_faces)}/{self.expected_face_count}) ---")

        if self.expected_face_count > 0 and len(self.collected_faces) >= self.expected_face_count:
            self.finalize_and_send_batch()

    # =========================================================
    # Baut das Dictionary für ein Gesicht
    # =========================================================
    def _build_person_dict(self, base_id, df_data, moon_data, face_image_path=None, source="real"):
        beschreibung = moon_data.get("description", "Keine Beschreibung gefunden.")
        if self.translator and self.target_lang == "de":
            beschreibung = self.translator.translate_text(beschreibung)

        person_dict = {
            "titel": f"ID: {str(base_id).upper()}",
            "geschlecht": df_data.get("Geschlecht", "Unbekannt"),
            "augen": "Braun",
            "stimmung": df_data.get("Emotion", "Neutral"),
            "alter": str(df_data.get("Alter", "N/A")),
            "gefahr": self._calculate_danger(df_data.get("Emotion", "Neutral")),
            "beschreibung": beschreibung,
            "source": source,
        }
        if face_image_path:
            person_dict["face_image_path"] = face_image_path
        return person_dict

    # =========================================================
    # Füllt fehlende Personen aus dem Pool auf
    # =========================================================
    def _append_pool_people(self, personen_daten):
        real_count = len(personen_daten)
        if real_count == 0 or real_count >= 4:
            return personen_daten
        if self.pool_loader is None:
            return personen_daten

        fehlende_slots = 4 - real_count
        pool_selection = self.pool_loader.get_pool_persons(fehlende_slots)
        for pool_person in pool_selection:
            personen_daten.append(
                self._build_person_dict(
                    base_id=pool_person.get("face_id", "pool"),
                    df_data=pool_person.get("deepface", {}),
                    moon_data=pool_person.get("moondream", {}),
                    face_image_path=pool_person.get("face_image_path"),
                    source=pool_person.get("source", "pool"),
                )
            )

        for index, person in enumerate(personen_daten, start=1):
            person["titel"] = f"ID: FACE{index}"

        return personen_daten

    # =========================================================
    # GUI oder Pool neu laden
    # =========================================================
    def reload_pool_loader(self):
        if self.pool_loader is None:
            return
        self.pool_loader.reload()

    # =========================================================
    # Danger-Score berechnen
    # =========================================================
    def _calculate_danger(self, emotion):
        danger_map = {
            "Wütend": "HOCH",
            "Beunruhigt": "MITTEL",
            "Angst": "MITTEL",
            "Ekel": "GERING"
        }
        return danger_map.get(emotion, "GERING")

    # =========================================================
    # Fertige Daten an GUI senden
    # =========================================================
    def finalize_and_send_batch(self):
        if not self.collected_faces:
            return

        data_to_send = self._append_pool_people(list(self.collected_faces))
        print(f"\n🚀 ALL FACES READY! Sending batch of {len(data_to_send)} to GUI...")
        self.data_finalized.emit("BATCH", data_to_send)

        time.sleep(2.0)
        self.cleanup_folders()

    # =========================================================
    # Ordner und internen Speicher zurücksetzen
    # =========================================================
    def cleanup_folders(self):
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

        # Interner Speicher zurücksetzen
        self.seen_files.clear()
        self.results_cache.clear()
        self.collected_faces = []
        self.expected_face_count = 0

        print("✨ System reset and ready for next person.")