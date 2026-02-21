# pip install PyQt6 (Für Windows "py -m pip install PyQt6"

import sys
import os
from PyQt6.QtWidgets import (QApplication, QGraphicsView, QGraphicsScene,
                             QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QFrame)
from PyQt6.QtGui import QPixmap, QFont, QColor, QPainter
from PyQt6.QtCore import Qt

# --- DATEN-KONFIGURATION ---
# Wenn hier weniger als 4 Personen stehen, werden die übrigen Plätze automatisch freigelassen.
PERSONEN_DATEN = [
    {"titel": "PERSON 1", "geschlecht": "Männlich", "augen": "Braun", "stimmung": "Neutral", "alter": "32",
     "gefahr": "GERING"},
    {"titel": "PERSON 2", "geschlecht": "Weiblich", "augen": "Blau", "stimmung": "Beunruhigt", "alter": "27",
     "gefahr": "MITTEL"},
    {"titel": "PERSON 3", "geschlecht": "Divers", "augen": "Grün", "stimmung": "Aggressiv", "alter": "41",
     "gefahr": "EXTREM"},
    #{"titel": "PERSON 4", "geschlecht": "Männlich", "augen": "Grau", "stimmung": "Unbekannt", "alter": "55",
     #"gefahr": "HOCH"},
]


class PersonContainer(QFrame):
    """Container-Widget für eine Personenakte mit Foto, Stammdaten und Gefahrenstufe."""
    def __init__(self, daten, index):
        """Initialisiert den Container mit Personendaten und Index für die Fallakte."""
        super().__init__()
        # Container-Größe für bessere Lesbarkeit beibehalten
        self.setFixedSize(680, 400)

        # STYLE: Keine Rahmen, kein Hintergrund (wirkt wie auf Papier gedruckt)
        self.setStyleSheet("background: transparent; border: none; color: #1a1a1a;")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 5, 10, 5)
        main_layout.setSpacing(15)

        # 1. Überschrift: GRADUATE (Groß und markant)
        header = QLabel(daten['titel'])
        header.setFont(QFont("Graduate", 30, QFont.Weight.Bold))
        main_layout.addWidget(header)

        content_layout = QHBoxLayout()

        # --- LINKE SEITE (Daten & Foto) ---
        left_side = QVBoxLayout()

        # Foto-Rahmen etwas "analoger"
        img_placeholder = QLabel("FOTO")
        img_placeholder.setFixedSize(160, 180)
        img_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Nur ein ganz dünner grauer Rahmen, wie ein aufgeklebtes Bild
        img_placeholder.setStyleSheet("background-color: #e2e2e2; border: 1px solid #aaa;")

        # Daten-Text: Goudy Bookletter 1911
        stats_font = QFont("Goudy Bookletter 1911", 16)
        stats_text = (f"GESCHLECHT: {daten['geschlecht']}<br>"
                      f"AUGENFARBE: {daten['augen']}<br>"
                      f"STIMMUNG: {daten['stimmung']}<br>"
                      f"ALTER: {daten['alter']}")

        stats_label = QLabel(stats_text)
        stats_label.setFont(stats_font)
        stats_label.setStyleSheet("line-height: 120%;")

        left_side.addWidget(img_placeholder)
        left_side.addWidget(stats_label)
        left_side.addStretch()

        # --- DEZENTE TRENNLINIE ---
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        # Sehr schwache Linie, wirkt wie eine Falz oder Bleistiftlinie
        line.setStyleSheet("color: rgba(0, 0, 0, 40);")

        # --- RECHTE SEITE (Akte & Beschreibung) ---
        right_side = QVBoxLayout()

        # Akten-Nummer: Goudy Bookletter 1911 (Mittelgroß)
        akte_titel = QLabel(f"Fallakte Nr: 2026/02/XY-{index + 1}")
        akte_titel.setFont(QFont("Goudy Bookletter 1911", 24, QFont.Weight.Bold))

        beschreibung = QLabel(
            "Dies ist ein generierter Beispieltext für die Personenbeschreibung. "
            "Die Akte enthält alle relevanten Details zur Identität und den "
            "beobachteten Verhaltensmustern der Zielperson.")
        beschreibung.setFont(QFont("Goudy Bookletter 1911", 18))
        beschreibung.setWordWrap(True)
        beschreibung.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Gefahr: Goudy Bookletter 1911 (Groß)
        gefahr_label = QLabel(f"GEFAHRENSTUFE: {daten['gefahr']}")
        gefahr_label.setFont(QFont("Goudy Bookletter 1911", 18, QFont.Weight.Bold))
        gefahr_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)

        if daten['gefahr'] == "EXTREM":
            gefahr_label.setStyleSheet("color: #a00000;")  # Dunkles "Blutrot"

        right_side.addWidget(akte_titel)
        right_side.addWidget(beschreibung)
        right_side.addStretch()
        right_side.addWidget(gefahr_label)

        content_layout.addLayout(left_side, 35)
        content_layout.addWidget(line)
        content_layout.addLayout(right_side, 65)

        main_layout.addLayout(content_layout)


class ScalingAkteGUI(QGraphicsView):
    """Skalierbare Hauptansicht für die Aktenseite inklusive Hintergrund und Einträge."""
    def __init__(self):
        """Richtet Szene, Hintergrundgrafik und Render-Optionen ein."""
        super().__init__()
        self.scene = QGraphicsScene(0, 0, 1920, 1080)
        self.setScene(self.scene)

        bg_path = "pictures/Akte_V3.png"
        if os.path.exists(bg_path):
            pixmap = QPixmap(bg_path)
            self.bg_item = self.scene.addPixmap(pixmap)
            self.bg_item.setPixmap(pixmap.scaled(1920, 1080, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                                 Qt.TransformationMode.SmoothTransformation))
        else:
            self.scene.setBackgroundBrush(QColor("#f4e4bc"))  # Papierähnliches Beige

        self.setup_ui_elements()

        # Rendering-Optionen für maximale Textschärfe
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)

    def setup_ui_elements(self):
        """Platziert die Personen-Container an vordefinierten Positionen, sofern Daten vorhanden sind."""
        pos_list = [
            (230, 80),  # Oben Links
            (1000, 80),  # Oben Rechts
            (230, 560),  # Unten Links
            (1000, 560)  # Unten Rechts
        ]

        for i, pos in enumerate(pos_list):
            # Prüfen, ob für diesen Index Daten vorhanden sind
            if i < len(PERSONEN_DATEN):
                container = PersonContainer(PERSONEN_DATEN[i], i)
                proxy = self.scene.addWidget(container)
                proxy.setPos(pos[0], pos[1])

    def resizeEvent(self, event):
        """Skaliert die Szene proportional beim Resize des Viewports."""
        super().resizeEvent(event)
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScalingAkteGUI()
    window.showMaximized()
    sys.exit(app.exec())