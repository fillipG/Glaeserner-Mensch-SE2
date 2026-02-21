import sys
import os
from PyQt6.QtWidgets import (QApplication, QGraphicsView, QGraphicsScene,
                             QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QFrame, QGraphicsObject)
from PyQt6.QtGui import QPixmap, QFont, QColor, QPainter
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPropertyAnimation, pyqtProperty, QEasingCurve

# --- DATEN-KONFIGURATION ---
PERSONEN_DATEN = [
    {"titel": "PERSON 1", "geschlecht": "Männlich", "augen": "Braun", "stimmung": "Neutral", "alter": "32",
     "gefahr": "GERING"},
    {"titel": "PERSON 2", "geschlecht": "Weiblich", "augen": "Blau", "stimmung": "Beunruhigt", "alter": "27",
     "gefahr": "MITTEL"},
    {"titel": "PERSON 3", "geschlecht": "Divers", "augen": "Grün", "stimmung": "Aggressiv", "alter": "41",
     "gefahr": "EXTREM"},
]


def create_dummy_pixmap(color, text, size=(200, 200)):
    pixmap = QPixmap(*size)
    pixmap.fill(QColor(color))
    painter = QPainter(pixmap)
    painter.setPen(QColor("white"))
    painter.setFont(QFont("Arial", 20))
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, text)
    painter.end()
    return pixmap


class AnimatedGraphicsButton(QGraphicsObject):
    clicked = pyqtSignal()

    def __init__(self, img1_path, img2_path=None, scale=0.15, parent=None):
        super().__init__(parent)
        self.start_scale = scale
        self.can_toggle = img2_path is not None  # Prüfen, ob ein zweites Bild existiert

        # Bilder laden
        self.pixmap1 = QPixmap(img1_path) if os.path.exists(img1_path) else create_dummy_pixmap("gray", "BTN")
        if self.can_toggle:
            self.pixmap2 = QPixmap(img2_path) if os.path.exists(img2_path) else create_dummy_pixmap("blue", "TOGGLE")
        else:
            self.pixmap2 = self.pixmap1

        self.current_pixmap = self.pixmap1
        self.is_toggled = False

        self.setScale(self.start_scale)
        # Ursprung in die Mitte für korrektes Pulsieren
        self.setTransformOriginPoint(self.boundingRect().center())

        # Animation Setup
        self._animation = QPropertyAnimation(self, b"scaleFactor", self)
        self._animation.setDuration(150)  # Etwas langsamer für bessere Sichtbarkeit

    def boundingRect(self):
        return QRectF(self.current_pixmap.rect())

    def paint(self, painter, option, widget):
        painter.drawPixmap(0, 0, self.current_pixmap)

    @pyqtProperty(float)
    def scaleFactor(self):
        return self.scale()

    @scaleFactor.setter
    def scaleFactor(self, factor):
        self.setScale(factor)

    def mousePressEvent(self, event):
        self._animation.stop()
        self._animation.setEasingCurve(QEasingCurve.Type.OutQuad)
        self._animation.setEndValue(self.start_scale * 0.85)
        self._animation.start()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self._animation.stop()
        # OutBack sorgt für ein schönes Zurückschnellen auf die Originalgröße
        self._animation.setEasingCurve(QEasingCurve.Type.OutBack)
        self._animation.setEndValue(self.start_scale)
        self._animation.start()

        if self.can_toggle:
            self.toggle_image()

        self.clicked.emit()
        super().mouseReleaseEvent(event)

    def toggle_image(self):
        self.is_toggled = not self.is_toggled
        self.current_pixmap = self.pixmap2 if self.is_toggled else self.pixmap1
        self.update()


class PersonContainer(QFrame):
    def __init__(self, daten, index):
        super().__init__()
        self.setFixedSize(680, 400)
        self.setStyleSheet("background: transparent; border: none; color: #1a1a1a;")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 5, 10, 5)
        main_layout.setSpacing(15)

        header = QLabel(daten['titel'])
        header.setFont(QFont("Graduate", 30, QFont.Weight.Bold))
        main_layout.addWidget(header)

        content_layout = QHBoxLayout()
        left_side = QVBoxLayout()
        img_placeholder = QLabel("FOTO")
        img_placeholder.setFixedSize(160, 180)
        img_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_placeholder.setStyleSheet("background-color: #e2e2e2; border: 1px solid #aaa;")

        stats_font = QFont("Goudy Bookletter 1911", 16)
        stats_text = (f"GESCHLECHT: {daten['geschlecht']}<br>AUGENFARBE: {daten['augen']}<br>"
                      f"STIMMUNG: {daten['stimmung']}<br>ALTER: {daten['alter']}")
        stats_label = QLabel(stats_text)
        stats_label.setFont(stats_font)

        left_side.addWidget(img_placeholder)
        left_side.addWidget(stats_label)
        left_side.addStretch()

        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setStyleSheet("color: rgba(0, 0, 0, 40);")

        right_side = QVBoxLayout()
        akte_titel = QLabel(f"Fallakte Nr: 2026/02/XY-{index + 1}")
        akte_titel.setFont(QFont("Goudy Bookletter 1911", 24, QFont.Weight.Bold))
        beschreibung = QLabel("Beispieltext für die Personenbeschreibung. Die Akte enthält alle Details.")
        beschreibung.setFont(QFont("Goudy Bookletter 1911", 18))
        beschreibung.setWordWrap(True)

        gefahr_label = QLabel(f"GEFAHRENSTUFE: {daten['gefahr']}")
        gefahr_label.setFont(QFont("Goudy Bookletter 1911", 18, QFont.Weight.Bold))
        gefahr_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        if daten['gefahr'] == "EXTREM": gefahr_label.setStyleSheet("color: #a00000;")

        right_side.addWidget(akte_titel)
        right_side.addWidget(beschreibung)
        right_side.addStretch()
        right_side.addWidget(gefahr_label)

        content_layout.addLayout(left_side, 35)
        content_layout.addWidget(line)
        content_layout.addLayout(right_side, 65)
        main_layout.addLayout(content_layout)


class ScalingAkteGUI(QGraphicsView):
    def __init__(self):
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
            self.scene.setBackgroundBrush(QColor("#f4e4bc"))

        self.setup_ui_elements()
        self.setup_buttons()

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)

    def setup_ui_elements(self):
        pos_list = [(230, 80), (1000, 80), (230, 560), (1000, 560)]
        for i, pos in enumerate(pos_list):
            if i < len(PERSONEN_DATEN):
                container = PersonContainer(PERSONEN_DATEN[i], i)
                proxy = self.scene.addWidget(container)
                proxy.setPos(pos[0], pos[1])

    def setup_buttons(self):
        # Skalierung der Buttons
        button_scale = 0.1
        # Zentrale X-Achse für beide Buttons (rechts auf dem Papier)
        center_x = 1250

        # 1. Sprach-Button
        self.btn_language = AnimatedGraphicsButton(
            "pictures/change_language_german.png",
            "pictures/change_language_english.png",
            scale=button_scale
        )
        # X-Position berechnen: Mitte minus halbe Breite des skalierten Bildes
        w1 = self.btn_language.pixmap1.width() * button_scale
        # y=40 ist nah am oberen Rand der Szene
        self.btn_language.setPos((center_x - (w1 / 2)), 0)
        self.scene.addItem(self.btn_language)

        # 2. Reset-Button
        self.btn_reset = AnimatedGraphicsButton(
            "pictures/reset_button.png",
            scale=button_scale
        )
        w2 = self.btn_reset.pixmap1.width() * button_scale
        # Genau unter den Sprachbutton (y=160)
        self.btn_reset.setPos((center_x - (w2 / 2)+60), 160)
        self.scene.addItem(self.btn_reset)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScalingAkteGUI()
    window.showMaximized()
    sys.exit(app.exec())