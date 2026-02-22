import os
import sys
import cv2

from PyQt6.QtWidgets import (QApplication, QGraphicsView, QGraphicsScene,
                             QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QFrame, QGraphicsObject, QPushButton, QSlider)
from PyQt6.QtGui import QPixmap, QFont, QColor, QPainter, QImage
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPropertyAnimation, pyqtProperty, QEasingCurve, QTimer
from sketch import create_advanced_sketch
from service import TranslationService

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
    """Admin-Menü mit statischen Anzeige- und Slider-Elementen."""
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
    """Grafischer Button mit Toggle- und Klick-Animation."""
    clicked = pyqtSignal()

    def __init__(self, img1_path, img2_path=None, scale=0.15, parent=None):
        super().__init__(parent)
        self.start_scale = scale
        self.can_toggle = img2_path is not None

        self.pixmap1 = QPixmap(img1_path) if os.path.exists(img1_path) else create_dummy_pixmap("gray", "BTN 1")
        self.pixmap2 = QPixmap(img2_path) if img2_path and os.path.exists(img2_path) else self.pixmap1

        self.current_pixmap = self.pixmap1
        self.is_toggled = False

        self.setScale(self.start_scale)
        self.setTransformOriginPoint(self.boundingRect().center())

        self._animation = QPropertyAnimation(self, b"scaleFactor", self)
        self._animation.setDuration(150)

    def boundingRect(self):
        return QRectF(self.current_pixmap.rect())

    def paint(self, painter, option, widget):
        painter.drawPixmap(0, 0, self.current_pixmap)

    @pyqtProperty(float)
    def scaleFactor(self): return self.scale()

    @scaleFactor.setter
    def scaleFactor(self, factor): self.setScale(factor)

    def mousePressEvent(self, event):
        # Akzeptiere das Event, damit das Release-Event an dieses Objekt geht
        event.accept()
        self._animation.stop()
        self._animation.setEndValue(self.start_scale * 0.85)
        self._animation.start()
        # Debugging Print: Wenn das erscheint, wurde die Hardware-Ebene erreicht
        print("Button gedrückt (Hardware-Event)")

    def mouseReleaseEvent(self, event):
        self._animation.stop()
        self._animation.setEasingCurve(QEasingCurve.Type.OutBack)
        self._animation.setEndValue(self.start_scale)
        self._animation.start()

        if self.can_toggle:
            self.is_toggled = not self.is_toggled
            self.current_pixmap = self.pixmap2 if self.is_toggled else self.pixmap1
            self.update()

        # Signal senden
        self.clicked.emit()
        super().mouseReleaseEvent(event)


# --- PERSONEN CONTAINER ---
class PersonContainer(QFrame):
    """Container fuer Personenkarte inkl. Uebersetzungslogik der festen Labels."""
    def __init__(self, daten, index, language="de"):
        super().__init__()
        self.daten = daten
        self.index = index
        self.language = language
        self._last_description_source = None
        self.setFixedSize(680, 400)
        self.setStyleSheet("background: transparent; border: none; color: #1a1a1a;")
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

        # Initialer Platzhalter
        self.beschreibung = TypewriterLabel("Warte auf Daten...", interval=20)
        self.beschreibung.setFont(QFont("Goudy Bookletter 1911", 18))
        self.beschreibung.setWordWrap(True)

        gefahr_label = QLabel(self._build_gefahr_text(self.language))
        gefahr_label.setFont(QFont("Goudy Bookletter 1911", 18, QFont.Weight.Bold))
        gefahr_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        if daten['gefahr'] == "EXTREM": gefahr_label.setStyleSheet("color: #a00000;")
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

# --- HAUPT GUI ---
class ScalingAkteGUI(QGraphicsView):
    """Haupt-GUI inklusive Spracheinstellung per config.yaml."""
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(0, 0, 1920, 1080)
        self.setScene(self.scene)

        self.video_cap = None
        self.video_item = None
        self.is_animating = False
        self.animation_speed = 0
        self.active_containers = []
        self.current_language = self._load_language_from_config()
        self.translator = TranslationService(target_lang=self.current_language)

        # Timer für das Scannen des "final" Ordners
        self.scan_timer = QTimer(self)
        self.scan_timer.timeout.connect(self.update_descriptions_from_files)
        self.scan_timer.start(2000)  # Scan alle 2 Sekunden

        self.admin_menu = AdminMenu(self)
        self.show_closed_folder()

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)

    def update_descriptions_from_files(self):
        """Scannt den 'final' Ordner und extrahiert die (ggf. mehrzeilige) 'description'."""
        folder_path = "final"
        if not os.path.exists(folder_path):
            return

        if not self.active_containers:
            return

        for i, container in enumerate(self.active_containers):
            file_name = f"face{i + 1}_moondream.yaml"
            file_path = os.path.join(folder_path, file_name)

            new_text = "Keine Daten gefunden. Akte ausstehend."

            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    description_lines = []
                    found_description = False

                    for line in lines:
                        clean_line = line.strip()

                        # Startpunkt finden
                        if clean_line.startswith("description:"):
                            found_description = True
                            # Inhalt nach dem Doppelpunkt in der ersten Zeile mitnehmen
                            first_part = line.split("description:", 1)[1].strip()
                            if first_part:
                                description_lines.append(first_part)
                            continue

                        # Wenn wir in der Description sind, weitersammeln bis zum naechsten Key
                        if found_description:
                            # Wenn die Zeile einen Doppelpunkt hat und am Anfang steht, ist es ein neuer Key
                            if ":" in clean_line and not line.startswith(" "):
                                break
                            description_lines.append(clean_line)

                    if description_lines:
                        new_text = " ".join(description_lines).strip()

                except Exception as e:
                    new_text = f"Fehler beim Lesen: {e}"

            if new_text != container._last_description_source:
                container._last_description_source = new_text
                translated_text = self.translator.translate_text(new_text) if self.translator else new_text
                if container.beschreibung.full_text != translated_text:
                    container.beschreibung.full_text = translated_text
                    container.beschreibung.start_typing()

    def _refresh_descriptions_for_language(self):
        for container in self.active_containers:
            if container._last_description_source:
                translated_text = self.translator.translate_text(container._last_description_source)
                if container.beschreibung.full_text != translated_text:
                    container.beschreibung.full_text = translated_text
                    container.beschreibung.start_typing()

    def show_closed_folder(self):
        self.scene.clear()
        self.active_containers = []  # Reset active containers
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

    def start_animation(self, checked=False, video_path="pictures/Akte_Animation.mp4", end_callback=None):
        # Support calls from QPushButton.clicked (passes a bool) and direct path calls.
        if isinstance(checked, (str, os.PathLike)):
            video_path = checked
            checked = False
        if not os.path.exists(video_path):
            (end_callback or self.show_open_folder)()
            return
        self._animation_end_callback = end_callback or self.show_open_folder
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
            q_img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            self.video_item.setPixmap(QPixmap.fromImage(q_img).scaled(1920, 1080))
            QTimer.singleShot(self.animation_speed, self.update_video_frame)
        else:
            self.is_animating = False
            self.video_cap.release()
            self.video_cap = None
            self.video_item = None
            QTimer.singleShot(100, self._animation_end_callback)

    def show_open_folder(self):
        self.scene.clear()
        bg = "pictures/Akte_V3.png"
        if os.path.exists(bg):
            self.scene.addPixmap(QPixmap(bg).scaled(1920, 1080, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                                    Qt.TransformationMode.SmoothTransformation))

        self.setup_ui_elements()
        self.setup_buttons()

        # Einmaliger manueller Aufruf zum Initialisieren der Texte aus Dateien
        self.update_descriptions_from_files()

        for container in self.active_containers:
            container.trigger_typing()

    def setup_ui_elements(self):
        self.active_containers = []
        pos_list = [(230, 80), (1000, 80), (230, 560), (1000, 560)]
        for i, pos in enumerate(pos_list):
            if i < len(PERSONEN_DATEN):
                container = PersonContainer(PERSONEN_DATEN[i], i, self.current_language)
                image_path = os.path.join("faces_yolo", f"face{i + 1}.jpg")
                if os.path.exists(image_path):
                    sketch_img = create_advanced_sketch(image_path)
                    container.set_sketch_image(sketch_img)
                proxy = self.scene.addWidget(container)
                proxy.setPos(pos[0], pos[1])
                proxy.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
                proxy.setZValue(1)
                self.active_containers.append(container)

    def setup_buttons(self):
        button_scale = 0.1
        center_x = 1250
        self.btn_language = AnimatedGraphicsButton("pictures/change_language_german.png",
                                                   "pictures/change_language_english.png", scale=button_scale)
        w1 = self.btn_language.pixmap1.width() * button_scale
        self.btn_language.setPos((center_x - (w1 / 2)), 0)
        self.btn_language.setZValue(100)
        self.scene.addItem(self.btn_language)
        if self.current_language == "en":
            self.btn_language.is_toggled = True
            self.btn_language.current_pixmap = self.btn_language.pixmap2
            self.btn_language.update()
        self.btn_language.clicked.connect(self.switch_language_logic)

        self.btn_reset = AnimatedGraphicsButton("pictures/reset_button.png", scale=button_scale)
        w2 = self.btn_reset.pixmap1.width() * button_scale
        self.btn_reset.setPos((center_x - (w2 / 2) + 60), 160)
        self.btn_reset.setZValue(100)
        self.scene.addItem(self.btn_reset)
        self.btn_reset.clicked.connect(self.reset_logic)

    def _load_language_from_config(self):
        """Liest die Sprache aus config.yaml, Standard ist 'de'."""
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            return "de"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("language:"):
                        value = stripped.split(":", 1)[1].strip().strip("\"'")
                        return value if value in {"de", "en"} else "de"
        except Exception:
            return "de"
        return "de"

    def _save_language_to_config(self, language):
        """Schreibt die Sprache in config.yaml, erzeugt den Key bei Bedarf."""
        config_path = "config.yaml"
        lines = []
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception:
                lines = []

        updated = False
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("language:"):
                prefix = line[:len(line) - len(stripped)]
                lines[i] = f"{prefix}language: {language}\n"
                updated = True
                break

        if not updated:
            lines.append(f"language: {language}\n")

        with open(config_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def _apply_language_to_containers(self, language):
        """Setzt die Sprache für alle aktiven Container."""
        for container in self.active_containers:
            container.apply_language(language)

    def switch_language_logic(self):
        """Wechselt die UI-Sprache und synchronisiert config.yaml."""
        if self.btn_language.is_toggled:
            self.current_language = "en"
            self._apply_language_to_containers("en")
            self._save_language_to_config("en")
            print("Status: Englisch")
        else:
            self.current_language = "de"
            self._apply_language_to_containers("de")
            self._save_language_to_config("de")
            print("Status: Deutsch")
        self.translator = TranslationService(target_lang=self.current_language)
        self._refresh_descriptions_for_language()

    def reset_logic(self):
        QTimer.singleShot(0, lambda: self.start_animation(
            "pictures/Akte_Animation_reverse.mp4",
            end_callback=self.show_closed_folder
        ))

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

class TypewriterLabel(QLabel):
    """Label mit Schreibmaschinen-Effekt."""
    finished = pyqtSignal()

    def __init__(self, full_text, interval=30, parent=None):
        super().__init__("", parent)
        self.full_text = full_text
        self.interval = interval
        self.current_index = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._type_char)

    def start_typing(self):
        self.setText("")
        self.current_index = 0
        self._timer.start(self.interval)

    def _type_char(self):
        if self.current_index < len(self.full_text):
            self.current_index += 1
            # Using slice to handle HTML tags better if needed
            self.setText(self.full_text[:self.current_index])
        else:
            self._timer.stop()
            self.finished.emit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScalingAkteGUI()
    window.showMaximized()
    sys.exit(app.exec())