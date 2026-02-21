import os
import sys
import cv2

from PyQt6.QtWidgets import (QApplication, QGraphicsView, QGraphicsScene,
                             QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QFrame, QGraphicsObject, QPushButton, QSlider)
from PyQt6.QtGui import QPixmap, QFont, QColor, QPainter, QImage
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPropertyAnimation, pyqtProperty, QEasingCurve, QTimer

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


# --- ADMIN MENÜ ---
class AdminMenu(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(450, 550)
        self.setStyleSheet("""
            background-color: rgba(45, 35, 25, 245);
            border: 3px solid #f4e4bc;
            border-radius: 15px;
            color: #f4e4bc;
        """)
        layout = QVBoxLayout(self)
        title = QLabel("ADMIN KONFIGURATION")
        title.setFont(QFont("Graduate", 22, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        for text in ["Scan-Tiefe", "KI-Überwachung", "Datendurchsatz", "System-Stabilität"]:
            layout.addSpacing(15)
            layout.addWidget(QLabel(text.upper()))
            s = QSlider(Qt.Orientation.Horizontal)
            s.setStyleSheet("QSlider::handle:horizontal { background: #f4e4bc; width: 18px; border-radius: 9px; } "
                            "QSlider::groove:horizontal { background: #3d2b1f; height: 10px; border-radius: 5px; }")
            layout.addWidget(s)

        layout.addStretch()
        layout.addWidget(QLabel("DRÜCKE 'E' ZUM VERLASSEN", alignment=Qt.AlignmentFlag.AlignCenter))
        self.hide()


# --- ANIMIERTE BUTTONS ---
class AnimatedGraphicsButton(QGraphicsObject):
    clicked = pyqtSignal()

    def __init__(self, img1_path, img2_path=None, scale=0.15, parent=None):
        super().__init__(parent)
        self.start_scale = scale
        self.can_toggle = img2_path is not None
        self.pixmap1 = QPixmap(img1_path) if os.path.exists(img1_path) else create_dummy_pixmap("gray", "BTN")
        self.pixmap2 = QPixmap(img2_path) if img2_path and os.path.exists(img2_path) else self.pixmap1
        self.current_pixmap = self.pixmap1
        self.is_toggled = False
        self.setScale(self.start_scale)
        self.setTransformOriginPoint(self.boundingRect().center())
        self._animation = QPropertyAnimation(self, b"scaleFactor", self)
        self._animation.setDuration(150)

    def boundingRect(self): return QRectF(self.current_pixmap.rect())

    def paint(self, painter, option, widget): painter.drawPixmap(0, 0, self.current_pixmap)

    @pyqtProperty(float)
    def scaleFactor(self): return self.scale()

    @scaleFactor.setter
    def scaleFactor(self, factor): self.setScale(factor)

    def mousePressEvent(self, event):
        self._animation.stop()
        self._animation.setEndValue(self.start_scale * 0.85)
        self._animation.start()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self._animation.stop()
        self._animation.setEasingCurve(QEasingCurve.Type.OutBack)
        self._animation.setEndValue(self.start_scale)
        self._animation.start()
        if self.can_toggle:
            self.is_toggled = not self.is_toggled
            self.current_pixmap = self.pixmap2 if self.is_toggled else self.pixmap1
            self.update()
        self.clicked.emit()
        super().mouseReleaseEvent(event)


# --- PERSONEN CONTAINER (DEIN ORIGINAL-DESIGN) ---
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


# --- HAUPT GUI ---
class ScalingAkteGUI(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(0, 0, 1920, 1080)
        self.setScene(self.scene)

        self.video_cap = None
        self.video_item = None
        self.is_animating = False
        self.animation_speed = 0  # ms Verzögerung zwischen Frames

        self.admin_menu = AdminMenu(self)
        self.show_closed_folder()

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)

    def show_closed_folder(self):
        self.scene.clear()
        self.is_animating = False
        path = "pictures/Akte_V1_Zu.png"
        if os.path.exists(path):
            self.scene.addPixmap(QPixmap(path).scaled(1920, 1080, Qt.AspectRatioMode.KeepAspectRatioByExpanding))

        self.btn_open = QPushButton("Mappe öffnen")
        self.btn_open.setFixedSize(300, 80)
        self.btn_open.setStyleSheet(
            "QPushButton { background-color: #3d2b1f; color: #f4e4bc; border: 3px solid #f4e4bc; border-radius: 15px; font-family: 'Graduate'; font-size: 24px; font-weight: bold; } QPushButton:hover { background-color: #5a4030; }")
        self.btn_open.clicked.connect(self.start_animation)
        proxy = self.scene.addWidget(self.btn_open)
        proxy.setPos(1350, 850)

    def start_animation(self):
        video_path = "pictures/Akte_Animation.mp4"
        if not os.path.exists(video_path):
            self.show_open_folder()
            return
        self.video_cap = cv2.VideoCapture(video_path)
        self.scene.clear()
        self.video_item = self.scene.addPixmap(QPixmap(1920, 1080))
        self.is_animating = True
        QTimer.singleShot(10, self.update_video_frame)

    def update_video_frame(self):
        if not self.is_animating or self.video_item is None or self.video_cap is None:
            return
        ret, frame = self.video_cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            # .copy() verhindert den Speicher-Crash (0xC0000409)
            q_img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            self.video_item.setPixmap(QPixmap.fromImage(q_img).scaled(1920, 1080))
            QTimer.singleShot(self.animation_speed, self.update_video_frame)
        else:
            self.is_animating = False
            self.video_cap.release()
            self.video_cap = None
            self.video_item = None
            # Zeit zum Aufräumen lassen vor dem Wechsel
            QTimer.singleShot(100, self.show_open_folder)

    def show_open_folder(self):
        self.scene.clear()
        bg = "pictures/Akte_V3.png"
        if os.path.exists(bg):
            self.scene.addPixmap(QPixmap(bg).scaled(1920, 1080, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                                    Qt.TransformationMode.SmoothTransformation))

        self.setup_ui_elements()
        self.setup_buttons()

    def setup_ui_elements(self):
        # Exakt deine gewünschten Positionen
        pos_list = [(230, 80), (1000, 80), (230, 560), (1000, 560)]
        for i, pos in enumerate(pos_list):
            if i < len(PERSONEN_DATEN):
                container = PersonContainer(PERSONEN_DATEN[i], i)
                proxy = self.scene.addWidget(container)
                proxy.setPos(pos[0], pos[1])

    def setup_buttons(self):
        # Exakt dein gewünschtes Button-Interface
        button_scale = 0.1
        center_x = 1250

        # 1. Sprach-Button
        self.btn_language = AnimatedGraphicsButton("pictures/change_language_german.png",
                                                   "pictures/change_language_english.png", scale=button_scale)
        w1 = self.btn_language.pixmap1.width() * button_scale
        self.btn_language.setPos((center_x - (w1 / 2)), 0)
        self.scene.addItem(self.btn_language)

        # 2. Reset-Button
        self.btn_reset = AnimatedGraphicsButton("pictures/reset_button.png", scale=button_scale)
        w2 = self.btn_reset.pixmap1.width() * button_scale
        self.btn_reset.setPos((center_x - (w2 / 2) + 60), 160)
        self.scene.addItem(self.btn_reset)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_E:
            if self.admin_menu.isVisible():
                self.admin_menu.hide()
            else:
                self.admin_menu.move((self.width() - 450) // 2, (self.height() - 550) // 2)
                self.admin_menu.show()
                self.admin_menu.raise_()
        super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScalingAkteGUI()
    window.showMaximized()
    sys.exit(app.exec())