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
        self._reset_button_pixmap_path = PATHS["reset_button"]
        self._reset_button_empty_path = PATHS["reset_button_empty"]
        self._reset_button_original_pixmap = None
        self._reset_countdown_item = None

        self.config_service = ConfigService(default_llm_value=LLM_OPTIONS[0]["value"])
        self.config = self._load_config()
        self.wait_time_file_closed = int(self.config.get("wait_time_file_closed", 3))
        self.reset_countdown_seconds = int(self.config.get("reset_countdown_seconds", 3))
        self.pipeline_timeout_seconds = int(self.config.get("pipeline_timeout_seconds", 30))
        self.animation_speed = int(self.config.get("animation_speed", 1))
        self.is_fullscreen = bool(self.config.get("fullscreen", True))
        self.developer_mode = bool(self.config.get("developer_mode", False))
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
        self.camera_pixmap_item = None
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

    def show_loading_indicator(self):
        """Zeigt ein Lade-Symbol je nach GUI-Zustand an und tauscht den Reset-Button aus."""
        self.loading_active = True
        self._start_pipeline_timeout()
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
        if not self.is_animating:
            self.show_closed_folder()

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
            self.camera_pixmap_item.setPixmap(pixmap)
        except Exception as e:
            print(f"on_camera_frame Fehler: {e}")

    def show_closed_folder(self):
        self._is_open = False
        self._set_state(GUIState.CLOSED)
        self.camera_pixmap_item = None

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

        self.wait_timer_item = CircularTimerItem(self.wait_time_file_closed, diameter=240)
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
        duration = max(1, int(self.wait_time_file_closed))
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
            QTimer.singleShot(100, self._animation_end_callback)

    def show_open_folder(self):
        self._is_open = True
        self._set_state(GUIState.OPEN)
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
        config = self.config_service.load()
        self._ensure_config_defaults(config)
        return config

    def _save_config(self):
        self.config_service.save(self.config)

    def _ensure_config_defaults(self, config):
        self.config_service.ensure_base_defaults(config)

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
            use_retinaface=True,
        )
        self._ensure_pipeline_entry(
            pipeline,
            model_id="fer",
            name="Emotionserkennung (FER)",
            enabled=False,
        )

    def _ensure_pipeline_entry(
        self,
        pipeline,
        model_id,
        name,
        enabled=False,
        prompt=None,
        show_preview=False,
        use_retinaface=None,
    ):
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
        if use_retinaface is not None:
            entry.setdefault("use_retinaface", bool(use_retinaface))

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
        self.admin_menu.wait_time_changed.connect(self._on_wait_time_changed)
        self.admin_menu.animation_speed_changed.connect(self._on_animation_speed_changed)
        self.admin_menu.pipeline_timeout_changed.connect(self._on_pipeline_timeout_changed)
        self.admin_menu.fullscreen_toggled.connect(self._on_fullscreen_toggled)
        self.admin_menu.developer_mode_toggled.connect(self._on_developer_mode_toggled)
        self.admin_menu.pool_enabled_changed.connect(self._on_pool_enabled_changed)
        self.admin_menu.pool_max_extra_changed.connect(self._on_pool_max_extra_changed)
        self.admin_menu.moondream_enabled_changed.connect(self._on_moondream_enabled)
        self.admin_menu.moondream_prompt_changed.connect(self._on_moondream_prompt)
        self.admin_menu.deepface_enabled_changed.connect(self._on_deepface_enabled)
        self.admin_menu.deepface_retinaface_changed.connect(self._on_deepface_retinaface_changed)
        self.admin_menu.fer_enabled_changed.connect(self._on_fer_enabled)
        self.admin_menu.llm_model_changed.connect(self._on_llm_model_changed)

    def _sync_admin_menu_with_config(self):
        moondream = self._get_pipeline_entry("moondream") or {}
        deepface = self._get_pipeline_entry("deepface") or {}
        fer = self._get_pipeline_entry("fer") or {}
        pool = self.config.get("pool", {})
        settings = {
            "wait_time_file_closed": self.config.get("wait_time_file_closed", 3),
            "animation_speed": self.config.get("animation_speed", 1),
            "pipeline_timeout_seconds": self.config.get("pipeline_timeout_seconds", 30),
            "fullscreen": self.config.get("fullscreen", True),
            "developer_mode": self.config.get("developer_mode", False),
            "pool_enabled": pool.get("enabled", True),
            "pool_max_extra_persons": pool.get("max_extra_persons", 3),
            "moondream_enabled": moondream.get("enabled", True),
            "moondream_prompt": moondream.get("prompt", ""),
            "deepface_enabled": deepface.get("enabled", False),
            "deepface_use_retinaface": deepface.get("use_retinaface", True),
            "fer_enabled": fer.get("enabled", False),
            "llm_model": self.config.get("llm_model", LLM_OPTIONS[0]["value"]),
        }
        self.admin_menu.apply_settings(settings)

    def _on_wait_time_changed(self, value):
        self.wait_time_file_closed = int(value)
        self._update_config_value("wait_time_file_closed", self.wait_time_file_closed)

    def _on_animation_speed_changed(self, value):
        self.animation_speed = int(value)
        self._update_config_value("animation_speed", self.animation_speed)

    def _on_pipeline_timeout_changed(self, value):
        self.pipeline_timeout_seconds = max(1, int(value))
        self._update_config_value("pipeline_timeout_seconds", self.pipeline_timeout_seconds)
        if self.loading_active:
            self._start_pipeline_timeout()

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
        QTimer.singleShot(0, lambda: self.start_animation(
            PATHS["close_animation"],
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
                PATHS["close_animation"],
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScalingAkteGUI()
    window.apply_window_state()
    sys.exit(app.exec())
