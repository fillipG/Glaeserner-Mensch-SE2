"""
Name: "ui_person_container.py"
Beschreibung: Baut die visuelle Personenkarte inklusive Texten, Bild und Sprachumschaltung.
Autor: Fillip Giffhorn und Dennis Penner (Textanimation)
"""

import cv2
import random

from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtGui import QPixmap, QFont, QImage
from PyQt6.QtCore import Qt
from .ui_typewriter import TypewriterLabel

TRANSLATIONS = {
    "de": {
        # Geschlecht
        "Man": "Mann",
        "Woman": "Frau",
        # Stimmung
        "angry": "wütend",
        "disgust": "angewidert",
        "fear": "ängstlich",
        "happy": "glücklich",
        "sad": "traurig",
        "surprise": "überrascht",
        "neutral": "neutral",
        # Gefahr
        "GERING": "GERING",
        "MITTEL": "MITTEL",
        "HOCH": "HOCH",
        "EXTREM": "EXTREM",
    },
    "en": {
        # Geschlecht
        "Mann": "Man",
        "Frau": "Woman",
        # Stimmung
        "angry": "angry",
        "disgust": "disgusted",
        "fear": "fearful",
        "happy": "happy",
        "sad": "sad",
        "surprise": "surprised",
        "neutral": "neutral",
        # Gefahr
        "GERING": "LOW",
        "MITTEL": "MEDIUM",
        "HOCH": "HIGH",
        "EXTREM": "EXTREME",
    }
}


class PersonContainer(QFrame):
    """Container fuer Personenkarte inkl. Uebersetzungslogik der festen Labels."""

    def __init__(self, daten, index, language="de", developer_mode=False):
        """
        Erstellt einen UI-Container fuer eine Person.
        :param daten: Personendaten fuer Anzeige und Texte.
        :param index: Position im aktuellen Datensatz.
        :param language: Aktuelle UI-Sprache.
        :param developer_mode: Aktiviert visuelle Debug-Hervorhebung.
        """
        super().__init__()
        self.setObjectName("person_container")
        self.daten = daten
        self.index = index
        self.language = language
        self.developer_mode = bool(developer_mode)
        self._last_description_source = None
        self._last_deepface_source = None
        self.setFixedSize(680, 480)
        self._apply_developer_mode_style()
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 5, 10, 5)
        main_layout.setSpacing(15)

        self.person_number = random.randint(100, 999)
        self.case_file_code = self._generate_case_file_code()

        self.header = TypewriterLabel(self._build_header_text(self.language), interval=60)
        self.header.setFont(QFont("Graduate", 30, QFont.Weight.Bold))
        main_layout.addWidget(self.header)

        content_layout = QHBoxLayout()
        left_side = QVBoxLayout()
        img_placeholder = QLabel("FOTO")
        img_placeholder.setFixedSize(160, 180)
        img_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_placeholder.setStyleSheet("background-color: #e2e2e2; border: 1px solid #aaa;")
        self.img_label = img_placeholder

        stats_font = QFont("Goudy Bookletter 1911", 16)
        stats_text = self._build_stats_text(self.language)
        self.stats_label = TypewriterLabel(stats_text, interval=25)
        self.stats_label.setFont(stats_font)

        left_side.addWidget(img_placeholder)
        left_side.addWidget(self.stats_label)
        left_side.addStretch()

        line = QFrame()
        line.setObjectName("separator_line")
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Plain)

        line.setLineWidth(2)
        line.setFixedHeight(300)
        line.setStyleSheet("QFrame#separator_line { background-color: rgba(0, 0, 0, 110); border: none; }")

        right_side = QVBoxLayout()
        akte_titel = TypewriterLabel(self._build_akte_title(self.language), interval=40)
        akte_titel.setFont(QFont("Goudy Bookletter 1911", 24, QFont.Weight.Bold))
        self.akte_titel = akte_titel

        self.beschreibung = TypewriterLabel("Warte auf Daten...", interval=20)
        self.beschreibung.setFont(QFont("Goudy Bookletter 1911", 18))
        self.beschreibung.setWordWrap(True)

        gefahr_label = QLabel(self._build_gefahr_text(self.language))
        gefahr_label.setFont(QFont("Goudy Bookletter 1911", 18, QFont.Weight.Bold))
        gefahr_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        if daten['gefahr'] == "EXTREM":
            gefahr_label.setStyleSheet("color: #a00000;")
        self.gefahr_label = gefahr_label

        right_side.addWidget(akte_titel)
        right_side.addWidget(self.beschreibung)
        right_side.addStretch()
        right_side.addWidget(gefahr_label)

        content_layout.addLayout(left_side, 35)
        content_layout.addWidget(line, 0, Qt.AlignmentFlag.AlignTop)
        content_layout.addLayout(right_side, 65)
        main_layout.addLayout(content_layout)

        self.typewriters = [self.header, self.stats_label, self.akte_titel, self.beschreibung]

    def set_sketch_image(self, sketch_img):
        """
        Setzt das Skizzenbild, das aus create_advanced_sketch() kommt.
        :param sketch_img: OpenCV-Bildmatrix in Grau- oder BGR-Format.
        """
        if sketch_img is None:
            return
        if len(sketch_img.shape) == 2:
            h, w = sketch_img.shape
            q_img = QImage(sketch_img.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        else:
            rgb = cv2.cvtColor(sketch_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            q_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self.img_label.setPixmap(
            QPixmap.fromImage(q_img).scaled(
                self.img_label.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _translate_value(self, value, language):
        """
        Uebersetzt einen Einzelwert ueber das Sprachmapping.
        :param value: Ursprungswert.
        :param language: Zielsprache als Sprachcode.
        :return: Uebersetzter oder unveraenderter Wert.
        """
        return TRANSLATIONS.get(language, {}).get(value, value)

    def _build_header_text(self, language):
        """
        Baut den Kopftext mit Personenkennung.
        :param language: Zielsprache als Sprachcode.
        :return: Vollstaendiger Headertext.
        """
        prefix = "PERSONENKENNZAHL" if language == "de" else "PERSON IDENTIFIER"
        return f"{prefix}: {self.person_number}"

    def _build_stats_text(self, language):
        """
        Erstellt den Statistikblock aus Geschlecht, Stimmung und Alter.
        :param language: Zielsprache als Sprachcode.
        :return: Mehrzeiliger Statistiktext.
        """
        labels = {
            "de": {"geschlecht": "GESCHLECHT", "stimmung": "STIMMUNG", "alter": "ALTER"},
            "en": {"geschlecht": "GENDER", "stimmung": "MOOD", "alter": "AGE"},
        }
        l = labels.get(language, labels["de"])

        geschlecht = self._translate_value(self.daten['geschlecht'], language)
        stimmung = self._translate_value(self.daten['stimmung'], language)

        return (f"{l['geschlecht']}: {geschlecht}\n"
                f"{l['stimmung']}: {stimmung}\n"
                f"{l['alter']}: {self.daten['alter']}")

    def _generate_case_file_code(self):
        """
        Erzeugt eine pseudozufaellige Fallakten-Nummer.
        :return: Formatierter Fallakten-Code.
        """
        hva_number = random.randint(80, 85)
        roman = random.choice(["IX", "IV"])
        serial = random.randint(1000, 9999)
        return f"HVA-{hva_number}/A-{roman}-{serial}"

    def _build_akte_title(self, language):
        """
        Baut den Titeltext fuer die Fallakte.
        :param language: Zielsprache als Sprachcode.
        :return: Vollstaendiger Titeltext.
        """
        prefix = "Fallakte" if language == "de" else "Case file"
        return f"{prefix}: {self.case_file_code}"

    def _build_gefahr_text(self, language):
        """
        Baut den Gefahrenstufen-Text.
        :param language: Zielsprache als Sprachcode.
        :return: Vollstaendiger Gefahren-Labeltext.
        """
        label = "GEFAHRENSTUFE" if language == "de" else "THREAT LEVEL"
        gefahr = self._translate_value(self.daten['gefahr'], language)
        return f"{label}: {gefahr}"

    def apply_language(self, language):
        """
        Aktualisiert nur die festen Labels (ohne Variablenwerte).
        :param language: Zielsprache als Sprachcode.
        """
        self.language = language
        self.header.full_text = self._build_header_text(language)
        self.header.start_typing()
        self.stats_label.full_text = self._build_stats_text(language)
        self.stats_label.start_typing()
        self.akte_titel.full_text = self._build_akte_title(language)
        self.akte_titel.start_typing()
        self.gefahr_label.setText(self._build_gefahr_text(language))

    def trigger_typing(self):
        """
        Startet die Schreibmaschinen-Animation fuer alle Textfelder.
        """
        for tw in self.typewriters:
            tw.start_typing()

    def update_stats_from_deepface(self, emotion=None, age=None, gender=None):
        """
        Aktualisiert die Statistikwerte mit neuen Deepface-Daten.
        :param emotion: Erkannte Emotion.
        :param age: Erkanntes Alter.
        :param gender: Erkanntes Geschlecht.
        """
        if emotion:
            self.daten["stimmung"] = str(emotion)
        if age:
            self.daten["alter"] = str(age)
        if gender:
            self.daten["geschlecht"] = str(gender)
        self.stats_label.full_text = self._build_stats_text(self.language)
        self.stats_label.start_typing()

    def _apply_developer_mode_style(self):
        """
        Setzt den Rahmenstil passend zum Developer-Mode.
        """
        if self.developer_mode:
            self.setStyleSheet(
                "QFrame#person_container { background: transparent; border: 2px solid #e65100; color: #1a1a1a; }"
            )
        else:
            self.setStyleSheet(
                "QFrame#person_container { background: transparent; border: none; color: #1a1a1a; }"
            )

    def set_developer_mode(self, enabled):
        """
        Aktiviert oder deaktiviert den Developer-Style.
        :param enabled: True aktiviert den Developer-Mode.
        """
        self.developer_mode = bool(enabled)
        self._apply_developer_mode_style()
