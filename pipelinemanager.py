"""
PipelineManager
----------------------------------
Diese Klasse regelt das Zusammenspiel der verschiedenen KI-Module. Da die KIs
(Deepface, Moondream, Ollama) unabhängig voneinander in Docker-Containern laufen,
dient dieser Manager als zentrale Sammelstelle für deren Ergebnisse.

Zuständigkeiten:
1. Überwachung der Ausgabe-Ordner auf neue Dateien.
2. Zuordnung der KI-Ergebnisse (Biometrie, Texte) zur richtigen Person.
3. Vollständigkeitsprüfung: Warten, bis alle KIs ihre Arbeit für eine Person beendet haben.
4. Auffüllen der Ergebnisse mit "Pool-Personen", damit die GUI immer vier Akten zeigt.
5. Vorbereitung und Übersetzung der Daten für die Anzeige in der GUI.

AUTOREN: Dennis Penner, FLorian Hoeft
"""

import os
import time
import yaml
import shutil
import random
from PyQt6.QtCore import QObject, pyqtSignal
from service import TranslationService
from path_service import get_paths
from constants import PipelineStage


class PipelineManager(QObject):
    """
    Signal-Schnittstelle zur Benutzeroberfläche (GUI):
    Dieses Signal wird gesendet, wenn die Datenverarbeitung abgeschlossen ist.
    Es überträgt einen Status-Text (zur Steuerung der GUI-Ansicht) und eine
    Liste mit den gesammelten Personendaten.
    """
    data_finalized = pyqtSignal(str, list)

    def __init__(self, config_data, pool_loader=None):
        super().__init__()
        self.paths = get_paths()
        self.pool_loader = pool_loader
        self.target_lang = "en"
        self.translator = None

        # INTERNER SPEICHER:
        # results_cache: Speichert Fragmente der KIs
        self.results_cache = {}
        self.collected_faces = []  # Liste der fertig verarbeiteten Personen-Objekte
        self.expected_face_count = 0  # Anzahl der Gesichter, die laut YOLO-Log zu erwarten sind
        self.last_logged_face_count = None
        self.seen_files = set()  # Verhindert Doppelt-Verarbeitung derselben Datei
        self.last_log_signature = None  # Zeitstempel der faces_log.yaml zur Erkennung neuer Durchläufe

        self.reload_config(config_data)

    def check_for_updates(self):
        """
        Hauptmethode (wird vom PipelineWorker-Thread aufgerufen).
        Prüft zuerst das Log von YOLO und scannt dann nach neuen KI-Ergebnisdateien.
        """
        try:
            log_name = "faces_log.yaml"
            log_path = os.path.join(self.watch_dir, log_name)

            # SCHRITT 1: Prüfen, wie viele Gesichter erkannt wurden
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    log_data = yaml.safe_load(f) or {}

                new_face_count = log_data.get("face_count", 0)
                # Signature prüft Dateialter und Inhalt -> erkennt neuen Foto-Vorgang
                log_signature = (os.path.getmtime(log_path), new_face_count)
                # Ein neuer Batch kann denselben face_count wie der vorherige haben.
                # Deshalb wird nicht nur auf die Anzahl, sondern auch auf die aktualisierte
                # Log-Datei selbst geprueft.
                if log_signature != self.last_log_signature:
                    # Ein neues Foto wurde gemacht -> Cache für neuen Durchgang leeren
                    self.results_cache.clear()
                    self.collected_faces.clear()
                    self.seen_files.clear()
                    self.last_log_signature = log_signature

                self.expected_face_count = new_face_count
                if self.last_logged_face_count != self.expected_face_count:
                    print(f"[LOG] Erwarte insgesamt {self.expected_face_count} Gesichter.")
                    self.last_logged_face_count = self.expected_face_count

                # Sonderfall: YOLO hat ausgelöst, aber kein Gesicht bestätigt
                if new_face_count <= 0:
                    self._handle_empty_batch()
                    return

            # SCHRITT 2: Neue Ergebnis-Dateien der KIs (z.B. face1_deepface.yaml) verarbeiten
            current_files = {f for f in os.listdir(self.watch_dir) if f.endswith(self.file_ext)}
            new_files = current_files - self.seen_files

            for file_name in new_files:
                if file_name == log_name:
                    continue
                # Kurze Pause, um sicherzustellen, dass die Datei fertig geschrieben wurde
                time.sleep(0.05)
                self.process_incoming_file(file_name)
                self.seen_files.add(file_name)

        except Exception as e:
            print(f"Fehler beim Ordner-Scan: {e}")

    def process_incoming_file(self, file_name):
        """
        Ordnet eine gefundene Datei einer Person zu und prüft auf Vollständigkeit.
        """
        try:
            name_no_ext = file_name.replace(self.file_ext, "")
            if "_" not in name_no_ext:
                return

            # Extrahiere Personen-ID und Modell-Name (z.B. 'face_0' und 'deepface')
            base_id, model_id = name_no_ext.rsplit("_", 1)

            if base_id not in self.results_cache:
                self.results_cache[base_id] = {}

            # Dateiinhalt in den Cache laden
            file_path = os.path.join(self.watch_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                self.results_cache[base_id][model_id] = data if isinstance(data, dict) else {"description": str(data)}

            # PRÜFUNG: Sind für diese ID alle benötigten KI-Analysen vorhanden?
            received = list(self.results_cache[base_id].keys())
            waiting_for = [m for m in self._get_effective_required_ids() if m not in received]

            print(f"[{base_id.upper()}] Empfangen: {model_id.upper()}")
            if not waiting_for:
                # Alle Daten für diese Person da -> zur Batch-Liste hinzufügen
                self.add_to_batch(base_id)

        except Exception as e:
            print(f"Fehler beim Verarbeiten von {file_name}: {e}")

    def _get_effective_required_ids(self):
        """Bestimmt basierend auf der Config, auf welche KI-Ergebnisse gewartet werden muss."""
        required_ids = list(self.required_ids)
        if self._is_ollama_enabled_in_config():
            return required_ids

        # Falls Ollama deaktiviert ist, wird Moondream als primäre Textquelle genutzt
        required_ids = [model_id for model_id in required_ids if model_id != PipelineStage.OLLAMA]
        if PipelineStage.MOONDREAM not in required_ids:
            required_ids.append(PipelineStage.MOONDREAM)
        return required_ids

    def add_to_batch(self, base_id):
        """Bereitet die Daten einer einzelnen Person final auf."""
        captured_data = self.results_cache.get(base_id, {})
        df_data = captured_data.get(PipelineStage.DEEPFACE, {})
        # Text kommt entweder von Ollama oder Moondream
        description_data = captured_data.get(PipelineStage.OLLAMA) or captured_data.get(PipelineStage.MOONDREAM, {})

        # Mapping der KI-Rohdaten auf das für die GUI benötigte Format
        person_dict = self._build_person_dict(base_id, df_data, description_data)
        self.collected_faces.append(person_dict)
        print(f"--- [COLLECTED] {base_id} ({len(self.collected_faces)}/{self.expected_face_count}) ---")

        # Wenn alle erkannten Gesichter verarbeitet sind -> Abgeschlossenes Paket an GUI senden
        if self.expected_face_count > 0 and len(self.collected_faces) >= self.expected_face_count:
            self.finalize_and_send_batch()

    def _build_person_dict(self, base_id, df_data, description_data, face_image_path=None, source="real"):
        """Erstellt das finale Daten-Objekt für eine Person (inkl. Übersetzung)."""
        beschreibung = description_data.get("description", "Keine Beschreibung gefunden.")

        # Falls die Sprache auf Deutsch gestellt ist, wird der KI-Text hier übersetzt
        if self.translator and self.target_lang == "de":
            beschreibung = self.translator.translate_text(beschreibung)

        person_dict = {
            "titel": f"ID: {str(base_id).upper()}",
            "geschlecht": df_data.get("Geschlecht", "Unbekannt"),
            "augen": "Braun",  # Dummy-Wert, da biometrisch schwer zu erfassen
            "stimmung": df_data.get("Emotion", "Neutral"),
            "alter": str(df_data.get("Alter", "N/A")),
            "gefahr": self._calculate_danger(df_data.get("Emotion", "Neutral")),
            "beschreibung": beschreibung,
            "source": source,
        }
        if face_image_path:
            person_dict["face_image_path"] = face_image_path
        return person_dict

    def _append_pool_people(self, personen_daten):
        """
        Füllt das Set auf 4 Personen auf.
        Ziel: In der Museumsanwendung sollen immer 4 Akten angezeigt werden, 
        auch wenn nur 1 Person vor der Kamera stand.
        """
        real_count = len(personen_daten)
        if real_count == 0 or real_count >= 4:
            return personen_daten
        if self.pool_loader is None:
            return personen_daten

        # Zufällige Fake-Personen aus dem vorinstallierten Pool laden
        fehlende_slots = 4 - real_count
        pool_selection = self.pool_loader.get_pool_persons(fehlende_slots)
        for pool_person in pool_selection:
            personen_daten.append(
                self._build_person_dict(
                    base_id=pool_person.get("face_id", "pool"),
                    df_data=pool_person.get("deepface", {}),
                    description_data=pool_person.get("ollama", {}),
                    face_image_path=pool_person.get("face_image_path"),
                    source=pool_person.get("source", "pool"),
                )
            )

        # IDs vereinheitlichen (FACE1, FACE2, etc.)
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

    def reload_config(self, config_data=None):
        if config_data is None:
            try:
                with open("config.yaml", "r", encoding="utf-8") as handle:
                    config_data = yaml.safe_load(handle) or {}
            except Exception as exc:
                print(f"Pipeline-Config konnte nicht neu geladen werden: {exc}")
                return

        self.target_lang = config_data.get("language", "en")
        if self.target_lang == "de":
            self.translator = TranslationService(target_lang='de')
        else:
            self.translator = None

        self.enabled_models = self._resolve_enabled_models(config_data)
        self.required_ids = [m["id"] for m in self.enabled_models]
        final_dir = self.paths["final"]

        # Die Pipeline beobachtet immer genau den Ordner des ersten finalen Modells.
        # Bei aktiviertem Ollama ist das der final-Ordner fuer OLLAMA/DEEPFACE,
        # bei deaktiviertem Ollama wird auf MOONDREAM im final-Ordner zurueckgefallen.
        # Wichtig: Der hier verwendete final-Fallback und pipeline[*].watch_dir muessen
        # konsistent bleiben. Wenn nur eine der beiden Quellen angepasst wird, laufen
        # Pipeline-Scan und Dateiausgabe auseinander.
        watch_dir = self.enabled_models[0]["watch_dir"] if self.enabled_models else final_dir
        # Relativen watch_dir aus der Config zu absolutem Pfad auflösen.
        # Falls er auf denselben Ordner wie final_dir zeigt (egal ob per base_dir
        # oder Einzelüberschreibung konfiguriert), immer das aufgelöste Path-Objekt nehmen.
        abs_watch = os.path.abspath(str(watch_dir).lstrip("./"))
        abs_final = os.path.abspath(os.fspath(final_dir))
        if abs_watch == abs_final:
            watch_dir = final_dir
        self.watch_dir = os.path.abspath(os.fspath(watch_dir))
        self.file_ext = ".yaml"
        os.makedirs(self.watch_dir, exist_ok=True)

        self.results_cache.clear()
        self.collected_faces.clear()
        self.expected_face_count = 0
        self.last_logged_face_count = None
        self.seen_files.clear()
        self.last_log_signature = None

        print("PIPELINE INITIALIZED")
        print(f"Waiting for models: {', '.join(self.required_ids).upper()}")
        print(f"Watching folder: {self.watch_dir}\n")

    def _resolve_enabled_models(self, config_data):
        pipeline = config_data.get("pipeline", [])
        if not isinstance(pipeline, list):
            return []

        # Nur explizit als final markierte Modelle duerfen einen Batch abschliessen.
        final_models = [
            cfg for cfg in pipeline
            if isinstance(cfg, dict) and cfg.get("enabled", False) and cfg.get("final_output", False)
        ]

        ollama_entry = next(
            (cfg for cfg in pipeline if isinstance(cfg, dict) and cfg.get("id") == PipelineStage.OLLAMA),
            None,
        )
        ollama_enabled = bool(ollama_entry.get("enabled", True)) if ollama_entry else False

        if ollama_enabled:
            return final_models

        moondream_entry = next(
            (cfg for cfg in pipeline if isinstance(cfg, dict) and cfg.get("id") == PipelineStage.MOONDREAM),
            None,
        )
        if moondream_entry and moondream_entry.get("enabled", False):
            # Ohne Ollama wird Moondream temporaer als finales Textmodell behandelt,
            # damit die Pipeline weiter auf eine Textdatei im final-Ordner warten kann.
            final_models = [cfg for cfg in final_models if cfg.get("id") != PipelineStage.OLLAMA]
            if not any(cfg.get("id") == PipelineStage.MOONDREAM for cfg in final_models):
                fallback_entry = dict(moondream_entry)
                fallback_entry["final_output"] = True
                fallback_entry["watch_dir"] = os.fspath(self.paths["final"])
                final_models.insert(0, fallback_entry)

        return final_models

    def _is_ollama_enabled_in_config(self):
        try:
            with open("config.yaml", "r", encoding="utf-8") as handle:
                config_data = yaml.safe_load(handle) or {}
        except Exception:
            return True

        pipeline = config_data.get("pipeline", [])
        if not isinstance(pipeline, list):
            return True

        for cfg in pipeline:
            if isinstance(cfg, dict) and cfg.get("id") == PipelineStage.OLLAMA:
                return bool(cfg.get("enabled", True))
        return True

    # =========================================================
    # Fertige Daten an GUI senden
    # =========================================================
    def _handle_empty_batch(self):
        if self.expected_face_count != 0:
            return
        print("[LOG] No faces detected in the captured image. Resetting batch.")
        self.data_finalized.emit("EMPTY", [])
        self.cleanup_folders()

    def finalize_and_send_batch(self):
        """Schließt den Vorgang ab und benachrichtigt die GUI."""
        if not self.collected_faces:
            return

        # Mit Pool-Leuten auffüllen und senden
        data_to_send = self._append_pool_people(list(self.collected_faces))
        print(f"\nAlle Ergebnisse bereit! Sende Batch von {len(data_to_send)} Personen an GUI...")
        self.data_finalized.emit("BATCH", data_to_send)

        # Speicher für diese Personengruppe leeren
        self.collected_faces = []

    def _calculate_danger(self, emotion: str) -> str:
        """
        Leitet die Gefahrenstufe deterministisch aus der erkannten Emotion ab.
        Unbekannte oder fehlende Werte fallen auf MITTEL zurück.
        """
        mapping = {
            "happy": "GERING",
            "neutral": "MITTEL",
            "surprise": "MITTEL",
            "surprised": "MITTEL",
            "sad": "HOCH",
            "fear": "HOCH",
            "fearful": "HOCH",
            "disgust": "HOCH",
            "disgusted": "HOCH",
            "angry": "EXTREM",
        }
        return mapping.get(str(emotion).lower().strip(), "MITTEL")

    def cleanup_folders(self):
        """
        Löscht alle temporären Dateien nach Abschluss oder Reset.
        Dies verhindert, dass die Pipeline bei der nächsten Person fälschlicherweise 
        alte KI-Ergebnisse einliest.
        """
        cleanup_targets = [
            os.path.abspath(os.fspath(self.watch_dir)),
            os.path.abspath(os.fspath(self.paths["ollama_inbox"])),
        ]

        for folder in cleanup_targets:
            print()
            print(f"Cleaning up: {os.path.basename(folder)}")

            if not os.path.exists(folder):
                continue

            # Die Ollama-Inbox wird bewusst mitgeleert, damit alte Zwischenfiles
            # keinen neuen Batch faelschlich als bereits fertig aussehen lassen.
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"[SKIP] {filename} wird noch verwendet: {e}")

        self._reset_batch_state()
        print("System zurückgesetzt und bereit für neue Erfassung.")

    def _reset_batch_state(self):
        # Interner Speicher wird nach Batch-Ende getrennt vom Dateisystem geleert.
        # So kann die GUI fertige Ergebnisse noch anzeigen, bis bewusst zurueckgesetzt wird.
        self.seen_files.clear()
        self.results_cache.clear()
        self.collected_faces = []
        self.expected_face_count = 0
        self.last_logged_face_count = None
        self.last_log_signature = None
