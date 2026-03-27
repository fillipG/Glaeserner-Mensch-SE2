"""
Name: "description_repository.py"
Beschreibung: Liest Beschreibungs- und Deepface-Dateien aus dem finalen Ausgabeordner.
Autor: Fillip Giffhorn
"""

import os
import re
import yaml


class DescriptionRepository:
    """
    Kapselt Dateizugriffe fuer Beschreibungsdaten im final-Ordner.
    """

    def __init__(self, base_dir="General ordner/final"):
        """
        Initialisiert das Repository mit einem Basisverzeichnis.
        :param base_dir: Ordnerpfad mit YAML-Ergebnissen.
        """
        self.base_dir = base_dir

    def exists(self):
        """
        Prueft, ob das Basisverzeichnis existiert.
        :return: True, wenn das Verzeichnis vorhanden ist.
        """
        return os.path.exists(self.base_dir)

    def read_faces_log_count(self):
        """
        Liest die erwartete Anzahl echter Gesichter aus faces_log.yaml.
        :return: Integer >= 0 oder None bei fehlender/ungueltiger Datei.
        """
        data = self.read_yaml("faces_log.yaml")
        if not isinstance(data, dict):
            return None
        try:
            return max(0, int(data.get("face_count", 0)))
        except (TypeError, ValueError):
            return None

    def _current_batch_id(self):
        """
        Liest die aktuelle Batch-ID aus faces_log.yaml.
        :return: Batch-ID als String oder None.
        """
        data = self.read_yaml("faces_log.yaml")
        if not isinstance(data, dict):
            return None
        batch_id = str(data.get("batch_id", "")).strip()
        return batch_id or None

    def list_face_indices(self):
        """
        Ermittelt alle im final-Ordner vorhandenen face-Indizes.
        Beruecksichtigt aktuelle Batch-Dateien wie batch123_face1_deepface.yaml
        und faellt bei Bedarf auf das alte face1_deepface.yaml-Schema zurueck.
        """
        if not os.path.isdir(self.base_dir):
            return []

        indices = set()
        batch_id = self._current_batch_id()
        batch_pattern = None
        if batch_id:
            batch_pattern = re.compile(
                rf"^{re.escape(batch_id)}_face(\d+)_(?:deepface|ollama|moondream)\.yaml$",
                re.IGNORECASE,
            )

        for file_name in os.listdir(self.base_dir):
            match = batch_pattern.match(file_name) if batch_pattern else None
            if match is None:
                match = re.match(r"^face(\d+)_(?:deepface|ollama|moondream)\.yaml$", file_name, re.IGNORECASE)
            if match:
                indices.add(int(match.group(1)))
        return sorted(indices)

    def read_yaml(self, file_name):
        """
        Liest eine YAML-Datei relativ zum Basisverzeichnis.
        :return: Geladene Daten oder None bei Fehler/fehlender Datei.
        """
        file_path = os.path.join(self.base_dir, file_name)
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return None

    def read_moondream_description(self, index):
        """
        Liest die Moondream-Beschreibung einer Person.
        :param index: Personenindex ab 0.
        :return: Beschreibungstext oder None.
        """
        return self._read_description_for_current_batch(index, "moondream")

    def read_ollama_description(self, index):
        """
        Liest die Ollama-Beschreibung einer Person.
        :param index: Personenindex ab 0.
        :return: Beschreibungstext oder None.
        """
        return self._read_description_for_current_batch(index, "ollama")

    def read_deepface_data(self, index):
        """
        Liest Deepface-Metadaten einer Person aus YAML.
        :param index: Personenindex ab 0.
        :return: Dictionary mit Deepface-Daten oder None.
        """
        return self._read_yaml_for_current_batch(index, "deepface")

    def _read_yaml_for_current_batch(self, index, model_name):
        """
        Liest zuerst die YAML des aktuellen Batches und faellt bei Bedarf
        auf das alte faceN-Dateischema zurueck.
        """
        file_name = self._resolve_batch_file_name(index, model_name)
        if file_name:
            data = self.read_yaml(file_name)
            if data is not None:
                return data
        return self.read_yaml(f"face{index + 1}_{model_name}.yaml")

    def _read_description_for_current_batch(self, index, model_name):
        """
        Liest das description-Feld fuer aktuellen Batch oder Legacy-Dateinamen.
        """
        file_name = self._resolve_batch_file_name(index, model_name)
        if file_name:
            description = self._read_description_file(file_name)
            if description:
                return description
        return self._read_description_file(f"face{index + 1}_{model_name}.yaml")

    def _resolve_batch_file_name(self, index, model_name):
        """
        Baut den Dateinamen fuer den aktuell aktiven Batch auf.
        """
        batch_id = self._current_batch_id()
        if not batch_id:
            return None
        return f"{batch_id}_face{index + 1}_{model_name}.yaml"

    def _read_description_file(self, file_name):
        """
        Liest und validiert das Feld description aus einer YAML-Datei.
        :param file_name: Dateiname innerhalb des Basisverzeichnisses.
        :return: Bereinigter Beschreibungstext oder None.
        """
        data = self.read_yaml(file_name)
        if not isinstance(data, dict):
            return None
        description = data.get("description")
        if isinstance(description, str) and description.strip():
            return description.strip()
        return None

