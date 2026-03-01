import os
import sys
import cv2
import yaml
import time

from PyQt6.QtWidgets import (QApplication, QGraphicsView, QGraphicsScene,
                             QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QFrame, QGraphicsObject, QPushButton, QSlider, QCheckBox, QLineEdit, QScrollArea, QSizePolicy)
from PyQt6.QtGui import QPixmap, QFont, QColor, QPainter, QImage, QPen
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


class AnimatedGraphicsButton(QGraphicsObject):
    clicked = pyqtSignal()

    def __init__(self, image1_path, image2_path=None, scale=1.0, parent=None):
        super().__init__(parent)
        self.pixmap1 = QPixmap(image1_path)
        self.pixmap2 = QPixmap(image2_path) if image2_path else self.pixmap1
        self.current_pixmap = self.pixmap1
        self.scale = float(scale)
        self.is_toggled = False
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)

    def boundingRect(self):
        if self.current_pixmap.isNull():
            return QRectF(0, 0, 0, 0)
        return QRectF(0, 0, self.current_pixmap.width() * self.scale,
                      self.current_pixmap.height() * self.scale)

    def paint(self, painter, option, widget=None):
        if self.current_pixmap.isNull():
            return
        target = self.boundingRect()
        source = QRectF(self.current_pixmap.rect())
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawPixmap(target, self.current_pixmap, source)

    def mousePressEvent(self, event):
        if self.pixmap2 and self.pixmap2.cacheKey() != self.pixmap1.cacheKey():
            self.is_toggled = not self.is_toggled
            self.current_pixmap = self.pixmap2 if self.is_toggled else self.pixmap1
            self.update()
        self.clicked.emit()
        super().mousePressEvent(event)


class CircularTimerItem(QGraphicsObject):
    """Runder Countdown-Overlay mit modernem Ring-Design."""
    def __init__(self, duration_s, diameter=220, parent=None):
        super().__init__(parent)
        self.duration_s = max(1, int(duration_s))
        self.diameter = int(diameter)
        self.progress = 0.0  # 0.0 .. 1.0
        self.remaining_s = self.duration_s
        self.setZValue(200)

    def boundingRect(self):
        return QRectF(0, 0, self.diameter, self.diameter)

    def set_progress(self, progress, remaining_s):
        self.progress = max(0.0, min(1.0, float(progress)))
        self.remaining_s = max(0, int(round(remaining_s)))
        self.update()

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.boundingRect().adjusted(10, 10, -10, -10)

        # Hintergrund-Ring
        bg_pen = QPen(QColor(120, 96, 72, 160), 20)
        bg_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(bg_pen)
        painter.drawEllipse(rect)

        # Fortschritts-Ring
        fg_pen = QPen(QColor(244, 228, 188, 230), 20)
        fg_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(fg_pen)
        start_angle = 90 * 16
        span_angle = -int(360 * 16 * self.progress)
        painter.drawArc(rect, start_angle, span_angle)

        # Text
        painter.setPen(QColor(244, 228, 188))
        font = QFont("Graduate", 35, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(self.boundingRect(), Qt.AlignmentFlag.AlignCenter, f"{self.remaining_s}s")

class LoadingSpinnerItem(QGraphicsObject):
    """Einfacher, typischer Lade-Spinner (animierter Kreisbogen)."""
    def __init__(self, diameter=120, color=QColor(244, 228, 188, 230), direction=1, parent=None):
        super().__init__(parent)
        self.diameter = int(diameter)
        self._angle = 0
        self._color = color
        self._direction = 1 if direction >= 0 else -1
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)
        self.setZValue(200)

    def boundingRect(self):
        return QRectF(0, 0, self.diameter, self.diameter)

    def _tick(self):
        self._angle = (self._angle + (8 * self._direction)) % 360
        self.update()

    def stop(self):
        if self._timer.isActive():
            self._timer.stop()

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.boundingRect().adjusted(10, 10, -10, -10)
        pen = QPen(self._color, 14)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        start_angle = int(self._angle * 16)
        span_angle = int(-270 * 16)
        painter.drawArc(rect, start_angle, span_angle)


class ResetCountdownItem(QGraphicsObject):
    """Einfacher Countdown-Text ohne zusaetzliche Animation."""
    def __init__(self, diameter=80, color=QColor(80, 160, 255, 230), parent=None):
        super().__init__(parent)
        self.diameter = int(diameter)
        self.remaining = 0
        self._color = color
        self.setZValue(210)

    def boundingRect(self):
        return QRectF(0, 0, self.diameter, self.diameter)

    def set_remaining(self, remaining):
        self.remaining = int(max(0, remaining))
        self.update()

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.boundingRect()
        font_size = max(12, int(self.diameter * 0.45))
        painter.setFont(QFont("Graduate", font_size, QFont.Weight.Bold))
        painter.setPen(self._color)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"{self.remaining}")

# --- ADMIN MENÜ ---
class AdminMenu(QFrame):
    """Admin-Menue mit Anzeige- und Slider-Elementen."""
    wait_time_changed = pyqtSignal(int)
    animation_speed_changed = pyqtSignal(int)
    fullscreen_toggled = pyqtSignal(bool)
    developer_mode_toggled = pyqtSignal(bool)
    moondream_enabled_changed = pyqtSignal(bool)
    moondream_prompt_changed = pyqtSignal(str)
    deepface_enabled_changed = pyqtSignal(bool)
    fer_enabled_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("""
            background-color: rgba(45, 35, 25, 245);
            border: 3px solid #f4e4bc;
            border-radius: 15px;
            color: #f4e4bc;
        """)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(16, 16, 16, 16)
        outer_layout.setSpacing(8)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet(
            "QScrollArea { background: transparent; }"
            "QScrollBar:vertical { background: #3d2b1f; width: 10px; margin: 2px; border-radius: 5px; }"
            "QScrollBar::handle:vertical { background: #f4e4bc; min-height: 20px; border-radius: 5px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
        )

        self.content_widget = QWidget()
        layout = QVBoxLayout(self.content_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        self.scroll_area.setWidget(self.content_widget)
        outer_layout.addWidget(self.scroll_area)

        title = QLabel("ADMIN KONFIGURATION")
        title.setFont(QFont("Graduate", 22, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        layout.addWidget(self._section_title("ALLGEMEINE EINSTELLUNGEN"))
        general_box = self._create_group_box()
        general_layout = QVBoxLayout(general_box)

        self.wait_time_label = QLabel("Wartezeit (Sek.)")
        self.wait_time_value = QLabel("1 s")
        self.wait_time_slider = self._create_slider(1, 60)
        self.wait_time_slider.valueChanged.connect(self._on_wait_time_changed)
        general_layout.addLayout(self._slider_row(self.wait_time_label, self.wait_time_slider, self.wait_time_value))

        self.anim_speed_label = QLabel("Animationsgeschwindigkeit")
        self.anim_speed_value = QLabel("1")
        self.anim_speed_slider = self._create_slider(1, 10)
        self.anim_speed_slider.valueChanged.connect(self._on_animation_speed_changed)
        general_layout.addLayout(self._slider_row(self.anim_speed_label, self.anim_speed_slider, self.anim_speed_value))

        layout.addWidget(general_box)

        layout.addWidget(self._section_title("GRAFIK"))
        graphics_box = self._create_group_box()
        graphics_layout = QVBoxLayout(graphics_box)
        self.fullscreen_button = QPushButton("Vollbild: AUS")
        self.fullscreen_button.setCheckable(True)
        self.fullscreen_button.setStyleSheet(
            "QPushButton { background-color: #3d2b1f; color: #f4e4bc; border: 2px solid #f4e4bc; "
            "border-radius: 10px; padding: 8px 14px; font-size: 16px; font-weight: bold; }"
            "QPushButton:checked { background-color: #5a4030; }"
        )
        self.fullscreen_button.toggled.connect(self._on_fullscreen_toggled)
        graphics_layout.addWidget(self.fullscreen_button)

        self.developer_mode_button = QPushButton("Developer Mode: AUS")
        self.developer_mode_button.setCheckable(True)
        self.developer_mode_button.setStyleSheet(
            "QPushButton { background-color: #3d2b1f; color: #f4e4bc; border: 2px solid #f4e4bc; "
            "border-radius: 10px; padding: 8px 14px; font-size: 16px; font-weight: bold; }"
            "QPushButton:checked { background-color: #5a4030; }"
        )
        self.developer_mode_button.toggled.connect(self._on_developer_mode_toggled)
        graphics_layout.addWidget(self.developer_mode_button)
        layout.addWidget(graphics_box)

        layout.addWidget(self._section_title("KI-MODELLE"))
        models_box = self._create_group_box()
        models_layout = QVBoxLayout(models_box)

        self.moondream_enabled, self.moondream_prompt = self._create_model_block(
            models_layout, "Moondream", has_prompt=True
        )
        self.moondream_enabled.toggled.connect(self.moondream_enabled_changed)
        self.moondream_prompt.editingFinished.connect(self._on_moondream_prompt_changed)

        self.deepface_enabled, _ = self._create_model_block(models_layout, "Deepface")
        self.deepface_enabled.toggled.connect(self.deepface_enabled_changed)

        self.fer_enabled, _ = self._create_model_block(models_layout, "FER")
        self.fer_enabled.toggled.connect(self.fer_enabled_changed)

        layout.addWidget(models_box)

        layout.addStretch()
        layout.addWidget(QLabel("DRUECKE 'E' ZUM VERLASSEN", alignment=Qt.AlignmentFlag.AlignCenter))
        self.hide()

    def _section_title(self, text):
        label = QLabel(text)
        label.setFont(QFont("Graduate", 16, QFont.Weight.Bold))
        label.setStyleSheet("color: #f4e4bc;")
        return label

    def _create_group_box(self):
        box = QFrame()
        box.setStyleSheet(
            "QFrame { background-color: rgba(61, 43, 31, 200); border: 1px solid rgba(244, 228, 188, 120); "
            "border-radius: 12px; }"
        )
        box.setContentsMargins(10, 8, 10, 8)
        return box

    def _create_slider(self, min_val, max_val):
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(min_val, max_val)
        s.setStyleSheet(
            "QSlider::handle:horizontal { background: #f4e4bc; width: 18px; border-radius: 9px; } "
            "QSlider::groove:horizontal { background: #3d2b1f; height: 10px; border-radius: 5px; }"
        )
        return s

    def _slider_row(self, label, slider, value_label):
        row = QHBoxLayout()
        label.setMinimumWidth(220)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        value_label.setMinimumWidth(60)
        row.addWidget(label)
        row.addWidget(slider, 1)
        row.addWidget(value_label)
        return row

    def _create_model_block(self, parent_layout, title, has_prompt=False):
        block = QFrame()
        block.setStyleSheet(
            "QFrame { background-color: rgba(45, 35, 25, 220); border: 1px solid rgba(244, 228, 188, 90); "
            "border-radius: 10px; }"
        )
        layout = QVBoxLayout(block)
        header = QLabel(title)
        header.setFont(QFont("Graduate", 14, QFont.Weight.Bold))
        layout.addWidget(header)
        enabled_box = QCheckBox("Aktiviert")
        enabled_box.setStyleSheet("QCheckBox { font-size: 14px; }")
        layout.addWidget(enabled_box)
        prompt_edit = None
        if has_prompt:
            prompt_label = QLabel("Prompt")
            prompt_edit = QLineEdit()
            prompt_edit.setPlaceholderText("Beschreibungsprompt...")
            prompt_edit.setStyleSheet(
                "QLineEdit { background-color: #2b2018; color: #f4e4bc; border: 1px solid #f4e4bc; "
                "border-radius: 6px; padding: 6px; }"
            )
            layout.addWidget(prompt_label)
            layout.addWidget(prompt_edit)
        parent_layout.addWidget(block)
        return enabled_box, prompt_edit

    def _on_wait_time_changed(self, value):
        self.wait_time_value.setText(f"{value} s")
        self.wait_time_changed.emit(value)

    def _on_animation_speed_changed(self, value):
        self.anim_speed_value.setText(str(value))
        self.animation_speed_changed.emit(value)

    def _on_fullscreen_toggled(self, checked):
        self.fullscreen_button.setText("Vollbild: AN" if checked else "Vollbild: AUS")
        self.fullscreen_toggled.emit(checked)

    def _on_developer_mode_toggled(self, checked):
        self.developer_mode_button.setText("Developer Mode: AN" if checked else "Developer Mode: AUS")
        self.developer_mode_toggled.emit(checked)

    def _on_moondream_prompt_changed(self):
        if self.moondream_prompt is None:
            return
        self.moondream_prompt_changed.emit(self.moondream_prompt.text().strip())

    def apply_settings(self, settings):
        self._set_slider_value(self.wait_time_slider, settings.get("wait_time_file_closed", 3))
        self.wait_time_value.setText(f"{self.wait_time_slider.value()} s")
        self._set_slider_value(self.anim_speed_slider, settings.get("animation_speed", 1))
        self.anim_speed_value.setText(str(self.anim_speed_slider.value()))

        self._set_toggle_button(self.fullscreen_button, settings.get("fullscreen", True))
        self.fullscreen_button.setText("Vollbild: AN" if self.fullscreen_button.isChecked() else "Vollbild: AUS")

        self._set_toggle_button(self.developer_mode_button, settings.get("developer_mode", False))
        self.developer_mode_button.setText(
            "Developer Mode: AN" if self.developer_mode_button.isChecked() else "Developer Mode: AUS"
        )

        self._set_checkbox_value(self.moondream_enabled, settings.get("moondream_enabled", True))
        if self.moondream_prompt is not None:
            self._set_lineedit_value(self.moondream_prompt, settings.get("moondream_prompt", ""))
        self._set_checkbox_value(self.deepface_enabled, settings.get("deepface_enabled", False))
        self._set_checkbox_value(self.fer_enabled, settings.get("fer_enabled", False))

    def _set_slider_value(self, slider, value):
        slider.blockSignals(True)
        slider.setValue(int(value))
        slider.blockSignals(False)

    def _set_toggle_button(self, button, checked):
        button.blockSignals(True)
        button.setChecked(bool(checked))
        button.blockSignals(False)

    def _set_checkbox_value(self, checkbox, checked):
        checkbox.blockSignals(True)
        checkbox.setChecked(bool(checked))
        checkbox.blockSignals(False)

    def _set_lineedit_value(self, lineedit, text):
        lineedit.blockSignals(True)
        lineedit.setText(text)
        lineedit.blockSignals(False)

    def update_geometry(self, parent_size):
        max_w = int(parent_size.width() * 0.6)
        max_h = int(parent_size.height() * 0.85)
        min_w = 420
        min_h = 420
        menu_w = max(min_w, min(max_w, parent_size.width() - 40))
        menu_h = max(min_h, min(max_h, parent_size.height() - 40))
        self.resize(menu_w, menu_h)
        self.move((parent_size.width() - menu_w) // 2, (parent_size.height() - menu_h) // 2)

# --- PERSONEN CONTAINER ---
class PersonContainer(QFrame):
    """Container fuer Personenkarte inkl. Uebersetzungslogik der festen Labels."""
    def __init__(self, daten, index, language="de"):
        super().__init__()
        self.daten = daten
        self.index = index
        self.language = language
        self._last_description_source = None
        self._last_deepface_source = None
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

    def update_stats_from_deepface(self, emotion=None, age=None, gender=None):
        if emotion:
            self.daten["stimmung"] = str(emotion)
        if age:
            self.daten["alter"] = str(age)
        if gender:
            self.daten["geschlecht"] = str(gender)
        self.stats_label.full_text = self._build_stats_text(self.language)
        self.stats_label.start_typing()

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
        self._is_open = False
        self.person_data = list(PERSONEN_DATEN)
        self.loading_item = None
        self.loading_active = False
        self._reset_button_pixmap_path = "pictures/reset_button.png"
        self._reset_button_empty_path = "pictures/reset_button_empty.png"
        self._reset_button_original_pixmap = None
        self._reset_countdown_item = None

        self.config = self._load_config()
        self.wait_time_file_closed = int(self.config.get("wait_time_file_closed", 3))
        self.reset_countdown_seconds = int(self.config.get("reset_countdown_seconds", 3))
        self.animation_speed = int(self.config.get("animation_speed", 1))
        self.is_fullscreen = bool(self.config.get("fullscreen", True))
        self.developer_mode = bool(self.config.get("developer_mode", False))
        self.active_containers = []
        self.current_language = self.config.get("language", "de")
        self.translator = TranslationService(target_lang=self.current_language)

        self.wait_timer = QTimer(self)
        self.wait_timer.timeout.connect(self._update_wait_timer)
        self.wait_timer_item = None
        self._wait_start_time = None
        self._wait_duration_s = 0

        self._reset_countdown_timer = QTimer(self)
        self._reset_countdown_timer.timeout.connect(self._update_reset_countdown)
        self._reset_countdown_remaining = 0

        # Timer für das Scannen des "final" Ordners
        self.scan_timer = QTimer(self)
        self.scan_timer.timeout.connect(self.update_descriptions_from_files)
        self.scan_timer.start(2000)  # Scan alle 2 Sekunden

        self.admin_menu = AdminMenu(self)
        self._connect_admin_menu()
        self._sync_admin_menu_with_config()
        self.show_closed_folder()

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)

    def _normalize_person_data(self, personen_daten):
        if personen_daten is None:
            return []
        if isinstance(personen_daten, list):
            data = personen_daten
        else:
            try:
                data = list(personen_daten)
            except TypeError:
                data = []
        return data[:4]

    def show_loading_indicator(self):
        """Zeigt ein Lade-Symbol je nach GUI-Zustand an und tauscht den Reset-Button aus."""
        self.loading_active = True
        if self.loading_item is not None:
            if hasattr(self.loading_item, "stop"):
                self.loading_item.stop()
            self.loading_item.hide()
            self.scene.removeItem(self.loading_item)
            self.loading_item = None

        if self._is_open and hasattr(self, "btn_reset"):
            self._set_reset_button_loading(True)
            button_rect = self.btn_reset.boundingRect()
            diameter = max(30, int(min(button_rect.width(), button_rect.height()) * 0.7))
            self.loading_item = LoadingSpinnerItem(
                diameter=diameter,
                color=QColor(80, 160, 255, 230),
                direction=-1,
            )
            self.scene.addItem(self.loading_item)
            btn_pos = self.btn_reset.pos()
            self.loading_item.setPos(
                btn_pos.x() + (button_rect.width() - diameter) / 2,
                btn_pos.y() + (button_rect.height() - diameter) / 2,
            )
        else:
            diameter = 240
            self.loading_item = LoadingSpinnerItem(diameter=diameter)
            self.scene.addItem(self.loading_item)
            self.loading_item.setPos(1920 - diameter - 450, 1080 - diameter - 120)

    def hide_loading_indicator(self):
        """Beendet das Ladesymbol und stellt den Reset-Button wieder her."""
        self.loading_active = False
        if self.loading_item is not None:
            if hasattr(self.loading_item, "stop"):
                self.loading_item.stop()
            self.loading_item.hide()
            self.scene.removeItem(self.loading_item)
            self.loading_item = None
        self._set_reset_button_loading(False)

    def _set_reset_button_loading(self, is_loading):
        if not hasattr(self, "btn_reset"):
            return
        if is_loading:
            if self._reset_button_original_pixmap is None:
                self._reset_button_original_pixmap = self.btn_reset.current_pixmap
            if os.path.exists(self._reset_button_empty_path):
                empty = QPixmap(self._reset_button_empty_path)
                self.btn_reset.pixmap1 = empty
                self.btn_reset.pixmap2 = empty
                self.btn_reset.current_pixmap = empty
                self.btn_reset.update()
        else:
            if self._reset_button_original_pixmap is not None:
                self.btn_reset.pixmap1 = self._reset_button_original_pixmap
                self.btn_reset.pixmap2 = self._reset_button_original_pixmap
                self.btn_reset.current_pixmap = self._reset_button_original_pixmap
                self.btn_reset.update()
            self._reset_button_original_pixmap = None

    def handle_new_dataset(self, personen_daten):
        """Main-Controller: verarbeitet neue Datensaetze (inkl. Beschreibung) und steuert die Anzeige."""
        self.person_data = self._normalize_person_data(personen_daten)
        self.show_loading_indicator()
        if self.is_animating:
            self._animation_end_callback = self.show_open_folder
            return
        if self._is_open:
            self.show_flip_video()
        else:
            self.start_animation()

    def update_descriptions_from_files(self):
        """Scannt den 'final' Ordner und extrahiert die (ggf. mehrzeilige) 'description'."""
        folder_path = "General ordner/final"
        if not os.path.exists(folder_path):
            return

        if not self.active_containers:
            return

        for i, container in enumerate(self.active_containers):
            # Behalte eine per Datensatz gesetzte Beschreibung und ueberschreibe sie nicht.
            if container._last_description_source:
                continue
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

            deepface_name = f"face{i + 1}_deepface.yaml"
            deepface_path = os.path.join(folder_path, deepface_name)
            if os.path.exists(deepface_path):
                try:
                    with open(deepface_path, "r", encoding="utf-8") as f:
                        deepface_data = yaml.safe_load(f) or {}
                except Exception:
                    deepface_data = {}

                emotion = deepface_data.get("Emotion") or deepface_data.get("emotion")
                age = deepface_data.get("Alter") or deepface_data.get("alter")
                gender = deepface_data.get("Geschlecht") or deepface_data.get("geschlecht")
                deepface_signature = (emotion, age, gender)

                if deepface_signature != container._last_deepface_source and any(deepface_signature):
                    container._last_deepface_source = deepface_signature
                    container.update_stats_from_deepface(emotion=emotion, age=age, gender=gender)

    def _refresh_descriptions_for_language(self):
        if not self.active_containers:
            return
        for container in self.active_containers:
            source_text = container._last_description_source
            if not source_text:
                continue
            translated = self.translator.translate_text(source_text) if self.translator else source_text
            if container.beschreibung.full_text != translated:
                container.beschreibung.full_text = translated
                container.beschreibung.start_typing()

    def show_closed_folder(self):
        self._is_open = False
        self.scene.clear()
        self.active_containers = []  # Reset active containers
        self.is_animating = False
        if self.wait_timer.isActive():
            self.wait_timer.stop()
        path = "pictures/Akte_V1_Zu.png"
        if os.path.exists(path):
            self.scene.addPixmap(QPixmap(path).scaled(1920, 1080, Qt.AspectRatioMode.KeepAspectRatioByExpanding))

        self.wait_timer_item = CircularTimerItem(self.wait_time_file_closed, diameter=240)
        self.scene.addItem(self.wait_timer_item)
        self.wait_timer_item.hide()
        self.wait_timer_item.setPos(1920 - self.wait_timer_item.diameter - 450,
                                    1080 - self.wait_timer_item.diameter - 120)

        self.btn_open = QPushButton("Mappe öffnen")
        self.btn_open.setFixedSize(300, 80)
        self.btn_open.setStyleSheet(
            "QPushButton { background-color: #3d2b1f; color: #f4e4bc; border: 3px solid #f4e4bc; border-radius: 15px; font-family: 'Graduate'; font-size: 24px; font-weight: bold; } QPushButton:hover { background-color: #5a4030; }")
        self.btn_open.clicked.connect(self.show_animation_with_timer)
        proxy = self.scene.addWidget(self.btn_open)
        proxy.setPos(50, 50)

    def show_animation_with_timer(self):
        self._start_wait_timer()

    def _start_wait_timer(self):
        if self.developer_mode:
            self.start_animation()
            return
        duration = max(1, int(self.wait_time_file_closed))
        self._wait_duration_s = duration
        self._wait_start_time = time.perf_counter()
        if self.wait_timer_item is None:
            return
        self.wait_timer_item.set_progress(0.0, duration)
        self.wait_timer_item.show()
        if hasattr(self, "btn_open"):
            self.btn_open.setEnabled(False)
        self.wait_timer.start(33)

    def _update_wait_timer(self):
        if self._wait_start_time is None:
            return
        elapsed = time.perf_counter() - self._wait_start_time
        remaining = max(0.0, self._wait_duration_s - elapsed)
        progress = min(1.0, elapsed / float(self._wait_duration_s))
        if self.wait_timer_item is not None:
            self.wait_timer_item.set_progress(progress, remaining)
        if remaining <= 0:
            self.wait_timer.stop()
            if self.wait_timer_item is not None:
                self.wait_timer_item.hide()
            if hasattr(self, "btn_open"):
                self.btn_open.setEnabled(True)
            self.start_animation()

    def start_animation(self, checked=False, video_path="pictures/Akte_animation.mov", end_callback=None):
        # Support calls from QPushButton.clicked (passes a bool) and direct path calls.
        if isinstance(checked, (str, os.PathLike)):
            video_path = checked
            checked = False
        self.hide_loading_indicator()
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
        self._is_open = True
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

    def show_flip_video(self):
        """Spielt das Umblättern-Video ab und kehrt danach zur offenen Mappe zurück."""
        self.start_animation("pictures/Akte_umblaettern.mov", end_callback=self.show_open_folder)

    def setup_ui_elements(self):
        self.active_containers = []
        pos_list = [(230, 80), (1000, 80), (230, 560), (1000, 560)]
        for i, pos in enumerate(pos_list):
            if i < len(self.person_data):
                container = PersonContainer(self.person_data[i], i, self.current_language)
                description = self.person_data[i].get("beschreibung")
                if description:
                    container._last_description_source = description
                    translated = self.translator.translate_text(description) if self.translator else description
                    container.beschreibung.full_text = translated
                    container.beschreibung.start_typing()
                image_path = os.path.join("General ordner/faces_yolo", f"face{i + 1}.png")
                if os.path.exists(image_path):
                    sketch_img = create_advanced_sketch(image_path)
                    container.set_sketch_image(sketch_img)
                container.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
                proxy = self.scene.addWidget(container)
                proxy.setPos(pos[0], pos[1])
                proxy.setZValue(1)
                self.active_containers.append(container)

    def setup_buttons(self):
        button_scale = 0.1
        right_margin = 40
        top_margin = 80
        vertical_gap = 150
        self.btn_language = AnimatedGraphicsButton("pictures/change_language_german.png",
                                                   "pictures/change_language_english.png", scale=button_scale)
        w1 = self.btn_language.pixmap1.width() * button_scale
        self.btn_language.setPos(1920 - right_margin - w1, top_margin)
        self.btn_language.setZValue(100)
        self.scene.addItem(self.btn_language)
        if self.current_language == "en":
            self.btn_language.is_toggled = True
            self.btn_language.current_pixmap = self.btn_language.pixmap2
            self.btn_language.update()
        self.btn_language.clicked.connect(self.switch_language_logic)

        self.btn_reset = AnimatedGraphicsButton(self._reset_button_pixmap_path, scale=button_scale)
        w2 = self.btn_reset.pixmap1.width() * button_scale
        self.btn_reset.setPos(1920 - right_margin - w2, top_margin + vertical_gap)
        self.btn_reset.setZValue(100)
        self.scene.addItem(self.btn_reset)
        self.btn_reset.clicked.connect(self.reset_logic)

    def _load_config(self):
        """Laedt die gesamte config.yaml und setzt Defaults."""
        config_path = "config.yaml"
        config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
            except Exception:
                config = {}
        if not isinstance(config, dict):
            config = {}
        self._ensure_config_defaults(config)
        return config

    def _save_config(self):
        config_path = "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, sort_keys=False, allow_unicode=False)

    def _ensure_config_defaults(self, config):
        config.setdefault("language", "de")
        config.setdefault("wait_time_file_closed", 3)
        config.setdefault("reset_countdown_seconds", 3)
        config.setdefault("fullscreen", True)
        config.setdefault("developer_mode", False)

        pipeline = config.setdefault("pipeline", [])
        if not isinstance(pipeline, list):
            pipeline = []
            config["pipeline"] = pipeline

        self._ensure_pipeline_entry(
            pipeline,
            model_id="moondream",
            name="Visual Description (VLM)",
            enabled=True,
            prompt="Name the clothing and any accessories the person is wearing",
            show_preview=True,
        )
        self._ensure_pipeline_entry(
            pipeline,
            model_id="deepface",
            name="Emotionserkennung",
            enabled=False,
        )
        self._ensure_pipeline_entry(
            pipeline,
            model_id="fer",
            name="Emotionserkennung (FER)",
            enabled=False,
        )

    def _ensure_pipeline_entry(self, pipeline, model_id, name, enabled=False, prompt=None, show_preview=False):
        entry = next((p for p in pipeline if p.get("id") == model_id), None)
        if entry is None:
            entry = {
                "id": model_id,
                "name": name,
                "enabled": enabled,
                "watch_dir": "./final",
                "file_ext": ".yaml",
            }
            pipeline.append(entry)
        entry.setdefault("name", name)
        entry.setdefault("enabled", enabled)
        entry.setdefault("watch_dir", "./final")
        entry.setdefault("file_ext", ".yaml")
        if show_preview:
            entry.setdefault("show_preview", True)
        if prompt is not None:
            entry.setdefault("prompt", prompt)

    def _get_pipeline_entry(self, model_id):
        pipeline = self.config.setdefault("pipeline", [])
        return next((p for p in pipeline if p.get("id") == model_id), None)

    def _update_config_value(self, key, value):
        self.config[key] = value
        self._save_config()

    def _update_pipeline_value(self, model_id, key, value):
        entry = self._get_pipeline_entry(model_id)
        if entry is None:
            self._ensure_pipeline_entry(self.config.setdefault("pipeline", []), model_id, model_id, enabled=False)
            entry = self._get_pipeline_entry(model_id)
        if entry is None:
            return
        entry[key] = value
        self._save_config()

    def _connect_admin_menu(self):
        self.admin_menu.wait_time_changed.connect(self._on_wait_time_changed)
        self.admin_menu.animation_speed_changed.connect(self._on_animation_speed_changed)
        self.admin_menu.fullscreen_toggled.connect(self._on_fullscreen_toggled)
        self.admin_menu.developer_mode_toggled.connect(self._on_developer_mode_toggled)
        self.admin_menu.moondream_enabled_changed.connect(self._on_moondream_enabled)
        self.admin_menu.moondream_prompt_changed.connect(self._on_moondream_prompt)
        self.admin_menu.deepface_enabled_changed.connect(self._on_deepface_enabled)
        self.admin_menu.fer_enabled_changed.connect(self._on_fer_enabled)

    def _sync_admin_menu_with_config(self):
        moondream = self._get_pipeline_entry("moondream") or {}
        deepface = self._get_pipeline_entry("deepface") or {}
        fer = self._get_pipeline_entry("fer") or {}
        settings = {
            "wait_time_file_closed": self.config.get("wait_time_file_closed", 3),
            "animation_speed": self.config.get("animation_speed", 1),
            "fullscreen": self.config.get("fullscreen", True),
            "developer_mode": self.config.get("developer_mode", False),
            "moondream_enabled": moondream.get("enabled", True),
            "moondream_prompt": moondream.get("prompt", ""),
            "deepface_enabled": deepface.get("enabled", False),
            "fer_enabled": fer.get("enabled", False),
        }
        self.admin_menu.apply_settings(settings)

    def _on_wait_time_changed(self, value):
        self.wait_time_file_closed = int(value)
        self._update_config_value("wait_time_file_closed", self.wait_time_file_closed)

    def _on_animation_speed_changed(self, value):
        self.animation_speed = int(value)
        self._update_config_value("animation_speed", self.animation_speed)

    def _on_fullscreen_toggled(self, enabled):
        self._set_fullscreen(bool(enabled))
        self._update_config_value("fullscreen", bool(enabled))

    def _on_developer_mode_toggled(self, enabled):
        self.developer_mode = bool(enabled)
        self._update_config_value("developer_mode", self.developer_mode)

    def _on_moondream_enabled(self, enabled):
        self._update_pipeline_value("moondream", "enabled", bool(enabled))

    def _on_moondream_prompt(self, text):
        self._update_pipeline_value("moondream", "prompt", text)

    def _on_deepface_enabled(self, enabled):
        self._update_pipeline_value("deepface", "enabled", bool(enabled))

    def _on_fer_enabled(self, enabled):
        self._update_pipeline_value("fer", "enabled", bool(enabled))

    def _set_fullscreen(self, enabled):
        self.is_fullscreen = bool(enabled)
        if self.is_fullscreen:
            self.showMaximized()
        else:
            self.showNormal()

    def apply_window_state(self):
        if self.is_fullscreen:
            self.showMaximized()
        else:
            self.show()

    # --- SPRACHE ---
    def _load_language_from_config(self):
        """Liest die Sprache aus config.yaml, Standard ist 'de'."""
        return self.config.get("language", "de")

    def _save_language_to_config(self, language):
        """Schreibt die Sprache in config.yaml, erzeugt den Key bei Bedarf."""
        self._update_config_value("language", language)

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
        if self._start_reset_countdown():
            return
        QTimer.singleShot(0, lambda: self.start_animation(
            "pictures/Akte_animation_reverse.mov",
            end_callback=self.show_closed_folder
        ))

    def _start_reset_countdown(self):
        if not self._is_open or not hasattr(self, "btn_reset"):
            return False
        if self._reset_countdown_timer.isActive():
            return True
        self._set_reset_button_empty(True)
        seconds = max(1, int(self.reset_countdown_seconds))
        self._reset_countdown_remaining = seconds

        if self._reset_countdown_item is not None:
            self.scene.removeItem(self._reset_countdown_item)
            self._reset_countdown_item = None

        button_rect = self.btn_reset.boundingRect()
        diameter = max(30, int(min(button_rect.width(), button_rect.height()) * 0.7))
        self._reset_countdown_item = ResetCountdownItem(diameter=diameter)
        self.scene.addItem(self._reset_countdown_item)
        btn_pos = self.btn_reset.pos()
        self._reset_countdown_item.setPos(
            btn_pos.x() + (button_rect.width() - diameter) / 2,
            btn_pos.y() + (button_rect.height() - diameter) / 2,
        )
        self._reset_countdown_item.set_remaining(self._reset_countdown_remaining)
        self._reset_countdown_timer.start(1000)
        return True

    def _update_reset_countdown(self):
        self._reset_countdown_remaining -= 1
        if self._reset_countdown_item is not None:
            self._reset_countdown_item.set_remaining(self._reset_countdown_remaining)
        if self._reset_countdown_remaining <= 0:
            self._reset_countdown_timer.stop()
            self._clear_reset_countdown()
            QTimer.singleShot(0, lambda: self.start_animation(
                "pictures/Akte_animation_reverse.mov",
                end_callback=self.show_closed_folder
            ))

    def _clear_reset_countdown(self):
        if self._reset_countdown_item is not None:
            self.scene.removeItem(self._reset_countdown_item)
            self._reset_countdown_item = None
        self._set_reset_button_empty(False)

    def _set_reset_button_empty(self, is_empty):
        if not hasattr(self, "btn_reset"):
            return
        if is_empty:
            if self._reset_button_original_pixmap is None:
                self._reset_button_original_pixmap = self.btn_reset.current_pixmap
            if os.path.exists(self._reset_button_empty_path):
                empty = QPixmap(self._reset_button_empty_path)
                self.btn_reset.pixmap1 = empty
                self.btn_reset.pixmap2 = empty
                self.btn_reset.current_pixmap = empty
                self.btn_reset.update()
        else:
            if self._reset_button_original_pixmap is not None:
                self.btn_reset.pixmap1 = self._reset_button_original_pixmap
                self.btn_reset.pixmap2 = self._reset_button_original_pixmap
                self.btn_reset.current_pixmap = self._reset_button_original_pixmap
                self.btn_reset.update()
            self._reset_button_original_pixmap = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_E:
            if self.admin_menu.isVisible():
                self.admin_menu.hide()
            else:
                self._sync_admin_menu_with_config()
                self.admin_menu.update_geometry(self.size())
                self.admin_menu.show()
                self.admin_menu.raise_()
        if event.key() == Qt.Key.Key_U:
            neue_personen_liste = [
                {"titel": "PERSON 1", "geschlecht": "Männlich", "augen": "Braun", "stimmung": "Neutral", "alter": "32",
                 "gefahr": "GERING", "beschreibung": "Testbeschreibung Person 1."},
                {"titel": "PERSON 2", "geschlecht": "Weiblich", "augen": "Blau", "stimmung": "Beunruhigt",
                 "alter": "27", "gefahr": "MITTEL", "beschreibung": "Testbeschreibung Person 2."},
                {"titel": "PERSON 1", "geschlecht": "Männlich", "augen": "Braun", "stimmung": "Neutral", "alter": "32", "gefahr": "GERING", "beschreibung": "Testbeschreibung Person 1."},
                {"titel": "PERSON 2", "geschlecht": "Weiblich", "augen": "Blau", "stimmung": "Beunruhigt", "alter": "27", "gefahr": "MITTEL", "beschreibung": "Testbeschreibung Person 2."},
            ]
            self.handle_new_dataset(neue_personen_liste)
        if event.key() == Qt.Key.Key_L:
            if self.loading_active:
                self.hide_loading_indicator()
            else:
                self.show_loading_indicator()
        super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        if self.admin_menu.isVisible():
            self.admin_menu.update_geometry(self.size())

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
    window.apply_window_state()
    sys.exit(app.exec())

