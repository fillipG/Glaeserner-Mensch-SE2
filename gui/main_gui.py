import os
import sys
import cv2
import time

from PyQt6.QtWidgets import (QApplication, QGraphicsView, QGraphicsScene,
                             QLabel, QFrame, QPushButton)
from PyQt6.QtGui import QPixmap, QFont, QColor, QPainter, QImage, QPen
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, pyqtSlot
from sketch import create_advanced_sketch
from service import TranslationService
from .config_service import ConfigService
from .description_repository import DescriptionRepository
from .gui_constants import SCENE_WIDTH, SCENE_HEIGHT, PATHS
from .gui_state import GUIState
from .ui_admin_menu import AdminMenu
from .ui_person_container import PersonContainer
from .ui_widgets import AnimatedGraphicsButton, CircularTimerItem, LoadingSpinnerItem, ResetCountdownItem

# --- DATEN-KONFIGURATION ---
PERSONEN_DATEN = [
    {"titel": "PERSON 1", "geschlecht": "Männlich", "augen": "Braun", "stimmung": "Neutral", "alter": "32",
     "gefahr": "GERING"},
    {"titel": "PERSON 2", "geschlecht": "Weiblich", "augen": "Blau", "stimmung": "Beunruhigt", "alter": "27",
     "gefahr": "MITTEL"},
    {"titel": "PERSON 3", "geschlecht": "Divers", "augen": "Grün", "stimmung": "Aggressiv", "alter": "41",
     "gefahr": "EXTREM"},
]

LLM_OPTIONS = [
    {"label": "Ollama - llama3 (schlau)", "value": "llama3.2:1b"},
    {"label": "Ollama - gemma3 (schnell)", "value": "gemma3"},
    {"label": "Ollama - phi3 (klein)", "value": "phi3:3.8b"},
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


# --- HAUPT GUI ---
class ScalingAkteGUI(QGraphicsView):
    """Haupt-GUI inklusive Spracheinstellung per config.yaml."""
    folder_closed = pyqtSignal()  # Wird emittiert wenn closed_folder angezeigt wird -> YOLOWorker fortsetzen
    presence_monitoring_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(0, 0, SCENE_WIDTH, SCENE_HEIGHT)
        self.setScene(self.scene)

        self.video_cap = None
        self.video_item = None
        self.is_animating = False
        self._is_open = False
        self.state = GUIState.CLOSED
        self.person_data = list(PERSONEN_DATEN)
        self.loading_item = None
        self.loading_active = False
        self._last_person_present = True
        self._missed_presence_checks = 0
        self._auto_close_monitoring_enabled = False
        self._auto_close_monitoring_pending = False
        self._pending_close = None
        self._reset_button_warning_active = False
        self._reset_button_pixmap_path = PATHS["reset_button"]
        self._reset_button_empty_path = PATHS["reset_button_empty"]
        self._reset_button_original_pixmap = None
        self._reset_countdown_item = None

        self.config_service = ConfigService(default_llm_value=LLM_OPTIONS[0]["value"])
        self.config = self._load_config()
        self._apply_runtime_settings_from_config()
        self.active_containers = []
        self.current_language = self.config.get("language", "de")
        self.translator = TranslationService(target_lang=self.current_language)
        self.description_repo = DescriptionRepository(PATHS["final_dir"])

        self.wait_timer = QTimer(self)
        self.wait_timer.timeout.connect(self._update_wait_timer)
        self.wait_timer_item = None
        self._wait_start_time = None
        self._wait_duration_s = 0

        self._reset_countdown_timer = QTimer(self)
        self._reset_countdown_timer.timeout.connect(self._update_reset_countdown)
        self._reset_countdown_remaining = 0
        self._no_person_warning_timer = QTimer(self)
        self._no_person_warning_timer.timeout.connect(self._blink_no_person_warning)
        self.camera_pixmap_item = None
        self._last_camera_preview_pixmap = None
        self._pipeline_timeout_timer = QTimer(self)
        self._pipeline_timeout_timer.setSingleShot(True)
        self._pipeline_timeout_timer.timeout.connect(self._on_pipeline_timeout)

        # Timer für das Scannen des "final" Ordners
        #self.scan_timer = QTimer(self)
        #self.scan_timer.timeout.connect(self.update_descriptions_from_files)
        #self.scan_timer.start(2000)  # Scan alle 2 Sekunden

        self.admin_menu = AdminMenu(LLM_OPTIONS, self)
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

    def _set_state(self, state):
        self.state = state

    def _freeze_camera_preview(self):
        if self.camera_pixmap_item is None:
            return
        if self._last_camera_preview_pixmap is not None:
            self.camera_pixmap_item.setPixmap(self._last_camera_preview_pixmap)
            return
        placeholder = QPixmap(self._cam_display_w, self._cam_display_h)
        placeholder.fill(QColor("black"))
        self.camera_pixmap_item.setPixmap(placeholder)

    def show_loading_indicator(self):
        """Zeigt ein Lade-Symbol je nach GUI-Zustand an und tauscht den Reset-Button aus."""
        if self.state != GUIState.RESULTS_READY:
            self._set_state(GUIState.ANALYZING)
        self.loading_active = True
        self._auto_close_monitoring_enabled = False
        if self.state != GUIState.RESULTS_READY:
            self._auto_close_monitoring_pending = False
        self._start_pipeline_timeout()
        if not self._is_open:
            self._freeze_camera_preview()
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
            self.loading_item.setPos(SCENE_WIDTH - diameter - 450, SCENE_HEIGHT - diameter - 120)

    def hide_loading_indicator(self):
        """Beendet das Ladesymbol und stellt den Reset-Button wieder her."""
        self.loading_active = False
        if self._pipeline_timeout_timer.isActive():
            self._pipeline_timeout_timer.stop()
        if self.loading_item is not None:
            if hasattr(self.loading_item, "stop"):
                self.loading_item.stop()
            self.loading_item.hide()
            self.scene.removeItem(self.loading_item)
            self.loading_item = None
        self._set_reset_button_loading(False)

    def _start_pipeline_timeout(self):
        timeout_ms = max(1, int(self.pipeline_timeout_seconds)) * 1000
        self._pipeline_timeout_timer.start(timeout_ms)

    def _on_pipeline_timeout(self):
        if not self.loading_active:
            return
        if self.developer_mode:
            print(f"Pipeline timeout after {self.pipeline_timeout_seconds} seconds. Returning to closed folder.")
        self.hide_loading_indicator()
        self.close_folder(reason="pipeline_timeout")

    def _set_reset_button_loading(self, is_loading):
        button = getattr(self, "btn_reset", None)
        if button is None:
            return
        try:
            if is_loading:
                if self._reset_button_original_pixmap is None:
                    self._reset_button_original_pixmap = button.current_pixmap
                if os.path.exists(self._reset_button_empty_path):
                    empty = QPixmap(self._reset_button_empty_path)
                    button.pixmap1 = empty
                    button.pixmap2 = empty
                    button.current_pixmap = empty
                    button.update()
            else:
                if self._reset_button_original_pixmap is not None:
                    button.pixmap1 = self._reset_button_original_pixmap
                    button.pixmap2 = self._reset_button_original_pixmap
                    button.current_pixmap = self._reset_button_original_pixmap
                    button.update()
                self._reset_button_original_pixmap = None
        except RuntimeError:
            self._reset_button_original_pixmap = None

    def _set_reset_button_warning(self, is_warning):
        button = getattr(self, "btn_reset", None)
        if button is None:
            return
        try:
            if is_warning:
                self._reset_button_warning_active = True
                button.setOpacity(0.35 if button.opacity() >= 0.99 else 1.0)
            else:
                self._reset_button_warning_active = False
                button.setOpacity(1.0)
        except RuntimeError:
            self._reset_button_warning_active = False

    def _stop_no_person_timer(self):
        if self._no_person_warning_timer.isActive():
            self._no_person_warning_timer.stop()
        self._missed_presence_checks = 0
        self._set_reset_button_warning(False)

    def _get_auto_close_missed_check_limit(self):
        interval_ms = max(100, int(self.no_person_check_interval_ms))
        timeout_ms = max(5000, int(self.close_on_no_person_seconds) * 1000)
        return max(1, int(round(timeout_ms / float(interval_ms))))

    def _get_warning_start_missed_checks(self):
        interval_ms = max(100, int(self.no_person_check_interval_ms))
        warning_checks = max(1, int(round(5000 / float(interval_ms))))
        return max(0, self._get_auto_close_missed_check_limit() - warning_checks)

    def _update_no_person_warning_state(self):
        if not self._is_open or self.loading_active or self.is_animating:
            self._stop_no_person_timer()
            return

        if self._missed_presence_checks >= self._get_warning_start_missed_checks():
            if not self._no_person_warning_timer.isActive():
                self._set_reset_button_warning(True)
                self._no_person_warning_timer.start(400)
        else:
            if self._no_person_warning_timer.isActive():
                self._no_person_warning_timer.stop()
            self._set_reset_button_warning(False)

    def _blink_no_person_warning(self):
        if not self._is_open or self.loading_active or self.is_animating:
            self._stop_no_person_timer()
            return
        self._set_reset_button_warning(True)

    def close_folder(self, reason: str = "", animated: bool = True):
        if self.is_animating:
            self._pending_close = {"reason": reason, "animated": animated}
            return

        if self._is_open and animated:
            QTimer.singleShot(0, lambda: self.start_animation(
                PATHS["close_animation"],
                end_callback=self.show_closed_folder
            ))
            return

        self.show_closed_folder()

    @pyqtSlot(list)
    def handle_new_dataset(self, personen_daten):
        """Main-Controller: verarbeitet neue Datensaetze."""
        if not personen_daten:
            return

        self.person_data = self._normalize_person_data(personen_daten)
        self.show_loading_indicator()

        if self.is_animating:
            QTimer.singleShot(100, lambda: self.handle_new_dataset(personen_daten))
            return

        if self._is_open:
            self.show_flip_video()
        else:
            self.start_animation()

    @pyqtSlot(str, list)
    def handle_pipeline_result(self, status, personen_daten):
        if status == "EMPTY" or not personen_daten:
            self._auto_close_monitoring_pending = False
            self._auto_close_monitoring_enabled = False
            self.hide_loading_indicator()
            self.close_folder(reason="empty_result")
            return
        self._set_state(GUIState.RESULTS_READY)
        self._auto_close_monitoring_pending = True
        self.handle_new_dataset(personen_daten)

    def update_descriptions_from_files(self):
        """Scannt den 'final' Ordner und extrahiert die (ggf. mehrzeilige) 'description'."""
        if not self.description_repo.exists():
            return

        if not self.active_containers:
            return

        for i, container in enumerate(self.active_containers):
            if container._last_description_source:
                continue

            new_text = "Keine Daten gefunden. Akte ausstehend."
            description = self.description_repo.read_moondream_description(i)
            if description:
                new_text = description

            if new_text != container._last_description_source:
                container._last_description_source = new_text
                translated_text = self.translator.translate_text(new_text) if self.translator else new_text
                if container.beschreibung.full_text != translated_text:
                    container.beschreibung.full_text = translated_text
                    container.beschreibung.start_typing()

            deepface_data = self.description_repo.read_deepface_data(i) or {}
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

    def on_camera_frame(self, frame):
        """Slot – vom YOLOWorker via frame_ready-Signal aufgerufen."""
        if self.camera_pixmap_item is None:
            return
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            q_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(q_img).scaled(
                self._cam_display_w, self._cam_display_h,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self._last_camera_preview_pixmap = pixmap
            self.camera_pixmap_item.setPixmap(pixmap)
        except Exception as e:
            print(f"on_camera_frame Fehler: {e}")

    @pyqtSlot(bool)
    def on_person_presence_changed(self, is_present):
        self._last_person_present = bool(is_present)
        if is_present:
            self._stop_no_person_timer()
            return

        if (
            not self._is_open or
            self.is_animating or
            self.loading_active or
            not self._auto_close_monitoring_enabled or
            not self.close_on_no_person_enabled
        ):
            return

        self._missed_presence_checks += 1
        print(
            f"Auto-close check {self._missed_presence_checks}/"
            f"{self._get_auto_close_missed_check_limit()} missed"
        )
        self._update_no_person_warning_state()
        if self._missed_presence_checks < self._get_auto_close_missed_check_limit():
            return

        self._stop_no_person_timer()
        self.close_folder(reason="auto_close")

    def show_closed_folder(self):
        self._is_open = False
        self._set_state(GUIState.IDLE)
        self.camera_pixmap_item = None
        self._auto_close_monitoring_enabled = False
        self._auto_close_monitoring_pending = False
        self._stop_no_person_timer()

        self.scene.clear()
        self.active_containers = []
        self.is_animating = False
        if self.wait_timer.isActive():
            self.wait_timer.stop()
        path = PATHS["closed_folder"]
        if os.path.exists(path):
            self.scene.addPixmap(QPixmap(path).scaled(SCENE_WIDTH, SCENE_HEIGHT, Qt.AspectRatioMode.KeepAspectRatioByExpanding))

        self._cam_display_w = 800
        self._cam_display_h = 450
        cam_x = 20
        cam_y = (SCENE_HEIGHT - self._cam_display_h) // 2

        border = self.scene.addRect(cam_x - 3, cam_y - 3, self._cam_display_w + 6, self._cam_display_h + 6)
        border.setPen(QPen(QColor("#f4e4bc"), 3))
        border.setZValue(9)

        placeholder = QPixmap(self._cam_display_w, self._cam_display_h)
        placeholder.fill(QColor("black"))
        new_item = self.scene.addPixmap(placeholder)
        new_item.setPos(cam_x, cam_y)
        new_item.setZValue(10)
        QTimer.singleShot(0, lambda: setattr(self, "camera_pixmap_item", new_item))

        cam_label = QLabel("LIVE KAMERA")
        cam_label.setFont(QFont("Graduate", 14, QFont.Weight.Bold))
        cam_label.setStyleSheet("color: #f4e4bc; background: transparent;")
        cam_label_proxy = self.scene.addWidget(cam_label)
        cam_label_proxy.setPos(cam_x, cam_y - 35)
        cam_label_proxy.setZValue(11)

        self.wait_timer_item = CircularTimerItem(self.photo_delay, diameter=240)
        self.scene.addItem(self.wait_timer_item)
        self.wait_timer_item.hide()
        self.wait_timer_item.setPos(SCENE_WIDTH - self.wait_timer_item.diameter - 450,
                                    SCENE_HEIGHT - self.wait_timer_item.diameter - 120)

        self.btn_open = QPushButton("Mappe öffnen")
        self.btn_open.setFixedSize(300, 80)
        self.btn_open.setStyleSheet(
            "QPushButton { background-color: #3d2b1f; color: #f4e4bc; border: 3px solid #f4e4bc; border-radius: 15px; font-family: 'Graduate'; font-size: 24px; font-weight: bold; } QPushButton:hover { background-color: #5a4030; }")
        self.btn_open.clicked.connect(self.show_animation_with_timer)
        proxy = self.scene.addWidget(self.btn_open)
        proxy.setPos(50, 50)

        self.folder_closed.emit()

    def show_animation_with_timer(self):
        if self.is_animating or self.active_containers:
            return
        if self.wait_timer_item is None:
            self.start_animation()
            return
        try:
            self.wait_timer_item.isVisible()
        except RuntimeError:
            self.wait_timer_item = None
            self.start_animation()
            return
        self._start_wait_timer()

    def _start_wait_timer(self):
        if self.developer_mode:
            self.start_animation()
            return
        duration = max(1, int(self.photo_delay))
        self._wait_duration_s = duration
        self._wait_start_time = time.perf_counter()
        if self.wait_timer_item is None:
            self.start_animation()
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
            try:
                self.wait_timer_item.set_progress(progress, remaining)
            except RuntimeError:
                self.wait_timer_item = None
        if remaining <= 0:
            self.wait_timer.stop()
            if self.wait_timer_item is not None:
                try:
                    self.wait_timer_item.hide()
                except RuntimeError:
                    self.wait_timer_item = None
            if hasattr(self, "btn_open"):
                self.btn_open.setEnabled(True)
            self.start_animation()

    def start_animation(self, checked=False, video_path=PATHS["open_animation"], end_callback=None):
        if isinstance(checked, (str, os.PathLike)):
            video_path = checked
            checked = False
        if end_callback == self.show_open_folder:
            if self._is_open:
                self._set_state(GUIState.FLIPPING)
            else:
                self._set_state(GUIState.OPENING)
        self.hide_loading_indicator()
        if not os.path.exists(video_path):
            (end_callback or self.show_open_folder)()
            return
        self._animation_end_callback = end_callback or self.show_open_folder
        self.video_cap = cv2.VideoCapture(video_path)
        self.scene.clear()
        self.video_item = self.scene.addPixmap(QPixmap(SCENE_WIDTH, SCENE_HEIGHT))
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
            self.video_item.setPixmap(QPixmap.fromImage(q_img).scaled(SCENE_WIDTH, SCENE_HEIGHT))
            QTimer.singleShot(self.animation_speed, self.update_video_frame)
        else:
            self.is_animating = False
            self.video_cap.release()
            self.video_cap = None
            self.video_item = None
            end_callback = self._animation_end_callback
            QTimer.singleShot(100, end_callback)

            if self._pending_close:
                pending = self._pending_close
                self._pending_close = None
                QTimer.singleShot(
                    120,
                    lambda: self.close_folder(
                        reason=pending["reason"],
                        animated=pending["animated"],
                    ),
                )

    def show_open_folder(self):
        self._is_open = True
        self._set_state(GUIState.RESULTS_READY)
        self._stop_no_person_timer()
        self._auto_close_monitoring_enabled = bool(self._auto_close_monitoring_pending)
        self._auto_close_monitoring_pending = False
        if self._auto_close_monitoring_enabled and self.close_on_no_person_enabled:
            print("Auto-close monitoring active")
            self._set_state(GUIState.PRESENCE_MONITORING)
            self.presence_monitoring_requested.emit()
        self.scene.clear()
        bg = PATHS["open_folder"]
        if os.path.exists(bg):
            self.scene.addPixmap(QPixmap(bg).scaled(SCENE_WIDTH, SCENE_HEIGHT, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                                    Qt.TransformationMode.SmoothTransformation))

        self.setup_ui_elements()
        self.setup_buttons()

        self.update_descriptions_from_files()

        for container in self.active_containers:
            container.trigger_typing()

        if self._auto_close_monitoring_enabled and self.close_on_no_person_enabled and not self._last_person_present:
            self._missed_presence_checks = 0
            self._update_no_person_warning_state()

    def show_flip_video(self):
        """Spielt das Umblättern-Video ab und kehrt danach zur offenen Mappe zurück."""
        self._set_state(GUIState.FLIPPING)
        self.start_animation(PATHS["flip_animation"], end_callback=self.show_open_folder)

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
                image_path = self.person_data[i].get("face_image_path")
                if not image_path or not os.path.exists(image_path):
                    image_path = os.path.join(PATHS["sketch_dir"], f"face{i + 1}.png")
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
        self.btn_language = AnimatedGraphicsButton(PATHS["language_de"],
                                                   PATHS["language_en"], scale=button_scale)
        w1 = self.btn_language.pixmap1.width() * button_scale
        self.btn_language.setPos(SCENE_WIDTH - right_margin - w1, top_margin)
        self.btn_language.setZValue(100)
        self.scene.addItem(self.btn_language)
        if self.current_language == "en":
            self.btn_language.is_toggled = True
            self.btn_language.current_pixmap = self.btn_language.pixmap2
            self.btn_language.update()
        self.btn_language.clicked.connect(self.switch_language_logic)

        self.btn_reset = AnimatedGraphicsButton(self._reset_button_pixmap_path, scale=button_scale)
        w2 = self.btn_reset.pixmap1.width() * button_scale
        self.btn_reset.setPos(SCENE_WIDTH - right_margin - w2, top_margin + vertical_gap)
        self.btn_reset.setZValue(100)
        self.scene.addItem(self.btn_reset)
        self.btn_reset.clicked.connect(self.reset_logic)

    def _load_config(self):
        """Laedt die gesamte config.yaml und setzt Defaults."""
        return self.config_service.load()

    def _save_config(self):
        self.config_service.save(self.config)

    def _ensure_config_defaults(self, config):
        self.config_service.ensure_defaults(config)

    def _get_pipeline_entry(self, model_id):
        pipeline = self.config.setdefault("pipeline", [])
        return next((p for p in pipeline if p.get("id") == model_id), None)

    def _update_config_value(self, key, value):
        self.config[key] = value
        self._save_config()

    def _get_face_yolo_confidence(self):
        face_yolo_cfg = self.config.get("face_yolo", {})
        if isinstance(face_yolo_cfg, dict):
            try:
                return float(face_yolo_cfg.get("confidence", 0.5))
            except (TypeError, ValueError):
                pass
        try:
            return float(self.config.get("face_yolo_confidence", 0.5))
        except (TypeError, ValueError):
            return 0.5

    def _update_pipeline_value(self, model_id, key, value):
        entry = self._get_pipeline_entry(model_id)
        if entry is None:
            self.config_service.ensure_defaults(self.config)
            entry = self._get_pipeline_entry(model_id)
        if entry is None:
            return
        entry[key] = value
        self._save_config()

    def _update_pool_value(self, key, value):
        pool = self.config.setdefault("pool", {})
        if not isinstance(pool, dict):
            pool = {}
            self.config["pool"] = pool
        pool[key] = value
        self._save_config()
        self._reload_pool_settings()

    def _reload_pool_settings(self):
        pipeline = getattr(self, "_pipeline", None)
        if pipeline is not None and hasattr(pipeline, "request_pool_reload"):
            pipeline.request_pool_reload()

    def _connect_admin_menu(self):
        self.admin_menu.photo_delay_changed.connect(self._on_photo_delay_changed)
        self.admin_menu.close_on_no_person_enabled_changed.connect(self._on_close_on_no_person_enabled_changed)
        self.admin_menu.close_on_no_person_changed.connect(self._on_close_on_no_person_changed)
        self.admin_menu.animation_speed_changed.connect(self._on_animation_speed_changed)
        self.admin_menu.pipeline_timeout_changed.connect(self._on_pipeline_timeout_changed)
        self.admin_menu.face_yolo_confidence_changed.connect(self._on_face_yolo_confidence_changed)
        self.admin_menu.fullscreen_toggled.connect(self._on_fullscreen_toggled)
        self.admin_menu.developer_mode_toggled.connect(self._on_developer_mode_toggled)
        self.admin_menu.pool_enabled_changed.connect(self._on_pool_enabled_changed)
        self.admin_menu.pool_max_extra_changed.connect(self._on_pool_max_extra_changed)
        self.admin_menu.pool_cooldown_changed.connect(self._on_pool_cooldown_changed)
        self.admin_menu.moondream_enabled_changed.connect(self._on_moondream_enabled)
        self.admin_menu.moondream_prompt_changed.connect(self._on_moondream_prompt)
        self.admin_menu.deepface_enabled_changed.connect(self._on_deepface_enabled)
        self.admin_menu.deepface_retinaface_changed.connect(self._on_deepface_retinaface_changed)
        self.admin_menu.fer_enabled_changed.connect(self._on_fer_enabled)
        self.admin_menu.llm_model_changed.connect(self._on_llm_model_changed)
        self.admin_menu.reset_defaults_requested.connect(self._reset_admin_settings_to_defaults)

    def _sync_admin_menu_with_config(self):
        defaults = self.config_service.get_default_admin_settings()
        moondream = self._get_pipeline_entry("moondream") or {}
        deepface = self._get_pipeline_entry("deepface") or {}
        fer = self._get_pipeline_entry("fer") or {}
        pool = self.config.get("pool", {})
        settings = {
            "photo_delay": self.config.get("photo_delay", defaults["photo_delay"]),
            "close_on_no_person_enabled": self.config.get(
                "close_on_no_person_enabled",
                defaults["close_on_no_person_enabled"]
            ),
            "close_on_no_person_seconds": self.config.get(
                "close_on_no_person_seconds",
                defaults["close_on_no_person_seconds"]
            ),
            "animation_speed": self.config.get("animation_speed", defaults["animation_speed"]),
            "pipeline_timeout_seconds": self.config.get(
                "pipeline_timeout_seconds",
                defaults["pipeline_timeout_seconds"]
            ),
            "face_yolo_confidence": self._get_face_yolo_confidence(),
            "fullscreen": self.config.get("fullscreen", defaults["fullscreen"]),
            "developer_mode": self.config.get("developer_mode", defaults["developer_mode"]),
            "pool_enabled": pool.get("enabled", defaults["pool_enabled"]),
            "pool_max_extra_persons": pool.get("max_extra_persons", defaults["pool_max_extra_persons"]),
            "pool_cooldown_batches": pool.get("cooldown_batches", defaults["pool_cooldown_batches"]),
            "moondream_enabled": moondream.get("enabled", defaults["moondream_enabled"]),
            "moondream_prompt": moondream.get("prompt", defaults["moondream_prompt"]),
            "deepface_enabled": deepface.get("enabled", defaults["deepface_enabled"]),
            "deepface_use_retinaface": deepface.get(
                "use_retinaface",
                defaults["deepface_use_retinaface"]
            ),
            "fer_enabled": fer.get("enabled", defaults["fer_enabled"]),
            "llm_model": self.config.get("llm_model", defaults["llm_model"]),
        }
        self.admin_menu.apply_settings(settings)

    def _apply_runtime_settings_from_config(self):
        defaults = self.config_service.get_default_config()
        self.photo_delay = int(self.config.get("photo_delay", defaults["photo_delay"]))
        self.reset_countdown_seconds = int(
            self.config.get("reset_countdown_seconds", defaults["reset_countdown_seconds"])
        )
        self.close_on_no_person_enabled = bool(
            self.config.get("close_on_no_person_enabled", defaults["close_on_no_person_enabled"])
        )
        self.close_on_no_person_seconds = int(
            self.config.get("close_on_no_person_seconds", defaults["close_on_no_person_seconds"])
        )
        self.no_person_check_interval_ms = int(
            self.config.get("no_person_check_interval_ms", defaults["no_person_check_interval_ms"])
        )
        self.pipeline_timeout_seconds = int(
            self.config.get("pipeline_timeout_seconds", defaults["pipeline_timeout_seconds"])
        )
        self.face_yolo_confidence = self._get_face_yolo_confidence()
        self.animation_speed = int(self.config.get("animation_speed", defaults["animation_speed"]))
        self.is_fullscreen = bool(self.config.get("fullscreen", defaults["fullscreen"]))
        self.developer_mode = bool(self.config.get("developer_mode", defaults["developer_mode"]))

    def _reset_admin_settings_to_defaults(self):
        self.config_service.reset_admin_settings(self.config)
        self._save_config()
        # Laufzeitwerte und Admin-UI sofort neu synchronisieren, damit der Reset direkt sichtbar ist.
        self._apply_runtime_settings_from_config()
        self._set_fullscreen(self.config.get("fullscreen", True))
        self._stop_no_person_timer()
        if self.loading_active:
            self._start_pipeline_timeout()
        self._reload_pool_settings()
        self._sync_admin_menu_with_config()

    def _on_photo_delay_changed(self, value):
        self.photo_delay = int(value)
        self._update_config_value("photo_delay", self.photo_delay)

    def _on_close_on_no_person_enabled_changed(self, enabled):
        self.close_on_no_person_enabled = bool(enabled)
        self._update_config_value("close_on_no_person_enabled", self.close_on_no_person_enabled)
        if not self.close_on_no_person_enabled:
            self._stop_no_person_timer()

    def _on_close_on_no_person_changed(self, value):
        self.close_on_no_person_seconds = max(5, min(60, int(value)))
        self._update_config_value("close_on_no_person_seconds", self.close_on_no_person_seconds)
        if self.close_on_no_person_enabled and self._is_open and not self._last_person_present:
            self._missed_presence_checks = 0
            self._update_no_person_warning_state()

    def _on_animation_speed_changed(self, value):
        self.animation_speed = int(value)
        self._update_config_value("animation_speed", self.animation_speed)

    def _on_pipeline_timeout_changed(self, value):
        self.pipeline_timeout_seconds = max(1, int(value))
        self._update_config_value("pipeline_timeout_seconds", self.pipeline_timeout_seconds)
        if self.loading_active:
            self._start_pipeline_timeout()

    def _on_face_yolo_confidence_changed(self, value):
        self.face_yolo_confidence = max(0.10, min(0.90, float(value)))
        face_yolo_cfg = self.config.setdefault("face_yolo", {})
        if not isinstance(face_yolo_cfg, dict):
            face_yolo_cfg = {}
            self.config["face_yolo"] = face_yolo_cfg
        face_yolo_cfg["confidence"] = round(self.face_yolo_confidence, 2)
        if "face_yolo_confidence" in self.config:
            del self.config["face_yolo_confidence"]
        self._save_config()

    def _on_fullscreen_toggled(self, enabled):
        self._set_fullscreen(bool(enabled))
        self._update_config_value("fullscreen", bool(enabled))

    def _on_developer_mode_toggled(self, enabled):
        self.developer_mode = bool(enabled)
        self._update_config_value("developer_mode", self.developer_mode)

    def _on_pool_enabled_changed(self, enabled):
        self._update_pool_value("enabled", bool(enabled))

    def _on_pool_max_extra_changed(self, value):
        pool_value = max(0, min(3, int(value)))
        self._update_pool_value("max_extra_persons", pool_value)

    def _on_pool_cooldown_changed(self, value):
        self._update_pool_value("cooldown_batches", max(0, int(value)))

    def _on_moondream_enabled(self, enabled):
        self._update_pipeline_value("moondream", "enabled", bool(enabled))

    def _on_moondream_prompt(self, text):
        self._update_pipeline_value("moondream", "prompt", text)

    def _on_deepface_enabled(self, enabled):
        self._update_pipeline_value("deepface", "enabled", bool(enabled))

    def _on_deepface_retinaface_changed(self, enabled):
        self._update_pipeline_value("deepface", "use_retinaface", bool(enabled))

    def _on_fer_enabled(self, enabled):
        self._update_pipeline_value("fer", "enabled", bool(enabled))

    def _on_llm_model_changed(self, value):
        self._update_config_value("llm_model", value)

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
        self.close_folder(reason="manual")

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
            self.close_folder(reason="manual_countdown")

    def _clear_reset_countdown(self):
        if self._reset_countdown_item is not None:
            self.scene.removeItem(self._reset_countdown_item)
            self._reset_countdown_item = None
        self._set_reset_button_empty(False)

    def _set_reset_button_empty(self, is_empty):
        button = getattr(self, "btn_reset", None)
        if button is None:
            return
        try:
            if is_empty:
                if self._reset_button_original_pixmap is None:
                    self._reset_button_original_pixmap = button.current_pixmap
                if os.path.exists(self._reset_button_empty_path):
                    empty = QPixmap(self._reset_button_empty_path)
                    button.pixmap1 = empty
                    button.pixmap2 = empty
                    button.current_pixmap = empty
                    button.update()
            else:
                if self._reset_button_original_pixmap is not None:
                    button.pixmap1 = self._reset_button_original_pixmap
                    button.pixmap2 = self._reset_button_original_pixmap
                    button.current_pixmap = self._reset_button_original_pixmap
                    button.update()
                self._reset_button_original_pixmap = None
        except RuntimeError:
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScalingAkteGUI()
    window.apply_window_state()
    sys.exit(app.exec())
