import cv2

from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtGui import QPixmap, QFont, QImage
from PyQt6.QtCore import Qt
from .ui_typewriter import TypewriterLabel


class PersonContainer(QFrame):
    """Container fuer Personenkarte inkl. Uebersetzungslogik der festen Labels."""
    def __init__(self, daten, index, language="de", developer_mode=False):
        super().__init__()
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

        self.header = TypewriterLabel(daten['titel'], interval=60)
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
        line.setFrameShape(QFrame.Shape.VLine)
        line.setStyleSheet("color: rgba(0, 0, 0, 40);")

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
        content_layout.addWidget(line)
        content_layout.addLayout(right_side, 65)
        main_layout.addLayout(content_layout)

        self.typewriters = [self.header, self.stats_label, self.akte_titel, self.beschreibung]

    def set_sketch_image(self, sketch_img):
        """Setzt das Skizzenbild, das aus create_advanced_sketch() kommt."""
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

    def _build_stats_text(self, language):
        labels = {
            "de": {"geschlecht": "GESCHLECHT", "augen": "AUGENFARBE", "stimmung": "STIMMUNG", "alter": "ALTER"},
            "en": {"geschlecht": "GENDER", "augen": "EYE COLOR", "stimmung": "MOOD", "alter": "AGE"},
        }
        l = labels.get(language, labels["de"])
        return (f"{l['geschlecht']}: {self.daten['geschlecht']}\n"
                f"{l['augen']}: {self.daten['augen']}\n"
                f"{l['stimmung']}: {self.daten['stimmung']}\n"
                f"{l['alter']}: {self.daten['alter']}")

    def _build_akte_title(self, language):
        prefix = "Fallakte Nr" if language == "de" else "Case file No"
        return f"{prefix}: 2026/02/XY-{self.index + 1}"

    def _build_gefahr_text(self, language):
        label = "GEFAHRENSTUFE" if language == "de" else "THREAT LEVEL"
        return f"{label}: {self.daten['gefahr']}"

    def apply_language(self, language):
        """Aktualisiert nur die festen Labels (ohne Variablenwerte)."""
        self.language = language
        self.stats_label.full_text = self._build_stats_text(language)
        self.stats_label.start_typing()
        self.akte_titel.full_text = self._build_akte_title(language)
        self.akte_titel.start_typing()
        self.gefahr_label.setText(self._build_gefahr_text(language))

    def trigger_typing(self):
        for tw in self.typewriters:
            tw.start_typing()

    def update_stats_from_deepface(self, emotion=None, age=None, gender=None):
        if emotion:
            self.daten["stimmung"] = str(emotion)
        if age:
            self.daten["alter"] = str(age)
        if gender:
            self.daten["geschlecht"] = str(gender)
        self.stats_label.full_text = self._build_stats_text(self.language)
        self.stats_label.start_typing()

    def _apply_developer_mode_style(self):
        if self.developer_mode:
            self.setStyleSheet(
                "background: transparent; border: 3px solid #e65100; color: #1a1a1a;"
            )
        else:
            self.setStyleSheet("background: transparent; border: none; color: #1a1a1a;")

    def set_developer_mode(self, enabled):
        self.developer_mode = bool(enabled)
        self._apply_developer_mode_style()
