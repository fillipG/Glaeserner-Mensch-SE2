"""
Name: "description_repository.py"
Beschreibung: Liest Beschreibungs- und Deepface-Dateien aus dem finalen Ausgabeordner.
Autor: Fillip Giffhorn
"""

import os
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

    def read_moondream_description(self, index):
        """
        Liest die Moondream-Beschreibung einer Person.
        :param index: Personenindex ab 0.
        :return: Beschreibungstext oder None.
        """
        return self._read_description_file(f"face{index + 1}_moondream.yaml")

    def read_ollama_description(self, index):
        """
        Liest die Ollama-Beschreibung einer Person.
        :param index: Personenindex ab 0.
        :return: Beschreibungstext oder None.
        """
        return self._read_description_file(f"face{index + 1}_ollama.yaml")

    def read_deepface_data(self, index):
        """
        Liest Deepface-Metadaten einer Person aus YAML.
        :param index: Personenindex ab 0.
        :return: Dictionary mit Deepface-Daten oder None.
        """
        file_name = f"face{index + 1}_deepface.yaml"
        file_path = os.path.join(self.base_dir, file_name)
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return None

    def _read_description_file(self, file_name):
        """
        Liest und validiert das Feld description aus einer YAML-Datei.
        :param file_name: Dateiname innerhalb des Basisverzeichnisses.
        :return: Bereinigter Beschreibungstext oder None.
        """
        file_path = os.path.join(self.base_dir, file_name)
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            return None
        description = data.get("description")
        if isinstance(description, str) and description.strip():
            return description.strip()
        return None

