from PyQt6.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QSlider, QCheckBox, QLineEdit, QScrollArea, QSizePolicy, QComboBox)
from PyQt6.QtGui import QFont, QIntValidator
from PyQt6.QtCore import Qt, pyqtSignal


class AdminMenu(QFrame):
    """Admin-Menue mit Anzeige- und Slider-Elementen."""
    wait_time_changed = pyqtSignal(int)
    animation_speed_changed = pyqtSignal(int)
    pipeline_timeout_changed = pyqtSignal(int)
    fullscreen_toggled = pyqtSignal(bool)
    developer_mode_toggled = pyqtSignal(bool)
    pool_enabled_changed = pyqtSignal(bool)
    pool_max_extra_changed = pyqtSignal(int)
    moondream_enabled_changed = pyqtSignal(bool)
    moondream_prompt_changed = pyqtSignal(str)
    deepface_enabled_changed = pyqtSignal(bool)
    deepface_retinaface_changed = pyqtSignal(bool)
    fer_enabled_changed = pyqtSignal(bool)
    llm_model_changed = pyqtSignal(str)

    def __init__(self, llm_options=None, parent=None):
        super().__init__(parent)
        self.llm_options = llm_options or []
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            "background-color: rgba(45, 35, 25, 245);"
            "border: 3px solid #f4e4bc;"
            "border-radius: 15px;"
            "color: #f4e4bc;"
        )
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

        self.content_widget = QFrame()
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

        timeout_row = QHBoxLayout()
        timeout_label = QLabel("Pipeline Timeout (Sek.)")
        timeout_label.setMinimumWidth(220)
        self.pipeline_timeout_edit = QLineEdit()
        self.pipeline_timeout_edit.setValidator(QIntValidator(1, 9999, self))
        self.pipeline_timeout_edit.setPlaceholderText("z.B. 45")
        self.pipeline_timeout_edit.setStyleSheet(
            "QLineEdit { background-color: #2b2018; color: #f4e4bc; border: 1px solid #f4e4bc; "
            "border-radius: 6px; padding: 6px; }"
        )
        self.pipeline_timeout_edit.editingFinished.connect(self._on_pipeline_timeout_changed)
        timeout_row.addWidget(timeout_label)
        timeout_row.addWidget(self.pipeline_timeout_edit, 1)
        general_layout.addLayout(timeout_row)

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

        layout.addWidget(self._section_title("POOL"))
        pool_box = self._create_group_box()
        pool_layout = QVBoxLayout(pool_box)

        self.pool_enabled_button = QPushButton("Pool: AUS")
        self.pool_enabled_button.setCheckable(True)
        self.pool_enabled_button.setStyleSheet(
            "QPushButton { background-color: #3d2b1f; color: #f4e4bc; border: 2px solid #f4e4bc; "
            "border-radius: 10px; padding: 8px 14px; font-size: 16px; font-weight: bold; }"
            "QPushButton:checked { background-color: #5a4030; }"
        )
        self.pool_enabled_button.toggled.connect(self._on_pool_enabled_toggled)
        pool_layout.addWidget(self.pool_enabled_button)

        self.pool_max_extra_label = QLabel("Zusatzpersonen")
        self.pool_max_extra_value = QLabel("0")
        self.pool_max_extra_slider = self._create_slider(0, 3)
        self.pool_max_extra_slider.valueChanged.connect(self._on_pool_max_extra_changed)
        pool_layout.addLayout(
            self._slider_row(self.pool_max_extra_label, self.pool_max_extra_slider, self.pool_max_extra_value)
        )
        layout.addWidget(pool_box)

        layout.addWidget(self._section_title("KI-MODELLE"))
        models_box = self._create_group_box()
        models_layout = QVBoxLayout(models_box)

        self.moondream_enabled, self.moondream_prompt, _ = self._create_model_block(
            models_layout, "Moondream", has_prompt=True
        )
        self.moondream_enabled.toggled.connect(self.moondream_enabled_changed)
        self.moondream_prompt.editingFinished.connect(self._on_moondream_prompt_changed)

        self.deepface_enabled, _, deepface_layout = self._create_model_block(models_layout, "Deepface")
        self.deepface_enabled.toggled.connect(self.deepface_enabled_changed)
        self.deepface_retinaface = QCheckBox("Verbessertes Analysemodell (RetinaFace)")
        self.deepface_retinaface.setStyleSheet("QCheckBox { font-size: 14px; }")
        self.deepface_retinaface.toggled.connect(self.deepface_retinaface_changed)
        deepface_layout.addWidget(self.deepface_retinaface)

        self.fer_enabled, _, _ = self._create_model_block(models_layout, "FER")
        self.fer_enabled.toggled.connect(self.fer_enabled_changed)

        layout.addWidget(models_box)

        layout.addWidget(self._section_title("LLM AUSWAHL"))
        llm_box = self._create_group_box()
        llm_layout = QVBoxLayout(llm_box)
        llm_label = QLabel("LLM Modell")
        llm_label.setFont(QFont("Graduate", 14, QFont.Weight.Bold))
        self.llm_combo = QComboBox()
        self.llm_combo.addItems([opt["label"] for opt in self.llm_options])
        self.llm_combo.setStyleSheet(
            "QComboBox { background-color: #2b2018; color: #f4e4bc; border: 1px solid #f4e4bc; "
            "border-radius: 6px; padding: 6px; }"
            "QComboBox::drop-down { border: 0px; }"
        )
        self.llm_combo.currentTextChanged.connect(self._on_llm_changed)
        llm_layout.addWidget(llm_label)
        llm_layout.addWidget(self.llm_combo)
        layout.addWidget(llm_box)

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
        return enabled_box, prompt_edit, layout

    def _on_wait_time_changed(self, value):
        self.wait_time_value.setText(f"{value} s")
        self.wait_time_changed.emit(value)

    def _on_animation_speed_changed(self, value):
        self.anim_speed_value.setText(str(value))
        self.animation_speed_changed.emit(value)

    def _on_pipeline_timeout_changed(self):
        text = self.pipeline_timeout_edit.text().strip()
        if not text:
            return
        try:
            value = max(1, int(text))
        except ValueError:
            return
        self.pipeline_timeout_edit.setText(str(value))
        self.pipeline_timeout_changed.emit(value)

    def _on_fullscreen_toggled(self, checked):
        self.fullscreen_button.setText("Vollbild: AN" if checked else "Vollbild: AUS")
        self.fullscreen_toggled.emit(checked)

    def _on_developer_mode_toggled(self, checked):
        self.developer_mode_button.setText("Developer Mode: AN" if checked else "Developer Mode: AUS")
        self.developer_mode_toggled.emit(checked)

    def _on_pool_enabled_toggled(self, checked):
        self.pool_enabled_button.setText("Pool: AN" if checked else "Pool: AUS")
        self.pool_enabled_changed.emit(checked)

    def _on_pool_max_extra_changed(self, value):
        self.pool_max_extra_value.setText(str(value))
        self.pool_max_extra_changed.emit(value)

    def _on_moondream_prompt_changed(self):
        if self.moondream_prompt is None:
            return
        self.moondream_prompt_changed.emit(self.moondream_prompt.text().strip())

    def _on_llm_changed(self, text):
        label_to_value = {opt["label"]: opt["value"] for opt in self.llm_options}
        value = label_to_value.get(text, self.llm_options[0]["value"])
        self.llm_model_changed.emit(value)

    def apply_settings(self, settings):
        self._set_slider_value(self.wait_time_slider, settings.get("wait_time_file_closed", 3))
        self.wait_time_value.setText(f"{self.wait_time_slider.value()} s")
        self._set_slider_value(self.anim_speed_slider, settings.get("animation_speed", 1))
        self.anim_speed_value.setText(str(self.anim_speed_slider.value()))
        self._set_lineedit_value(self.pipeline_timeout_edit, str(settings.get("pipeline_timeout_seconds", 30)))

        self._set_toggle_button(self.fullscreen_button, settings.get("fullscreen", True))
        self.fullscreen_button.setText("Vollbild: AN" if self.fullscreen_button.isChecked() else "Vollbild: AUS")

        self._set_toggle_button(self.developer_mode_button, settings.get("developer_mode", False))
        self.developer_mode_button.setText(
            "Developer Mode: AN" if self.developer_mode_button.isChecked() else "Developer Mode: AUS"
        )

        self._set_toggle_button(self.pool_enabled_button, settings.get("pool_enabled", True))
        self.pool_enabled_button.setText("Pool: AN" if self.pool_enabled_button.isChecked() else "Pool: AUS")
        self._set_slider_value(self.pool_max_extra_slider, settings.get("pool_max_extra_persons", 3))
        self.pool_max_extra_value.setText(str(self.pool_max_extra_slider.value()))

        self._set_checkbox_value(self.moondream_enabled, settings.get("moondream_enabled", True))
        if self.moondream_prompt is not None:
            self._set_lineedit_value(self.moondream_prompt, settings.get("moondream_prompt", ""))
        self._set_checkbox_value(self.deepface_enabled, settings.get("deepface_enabled", False))
        self._set_checkbox_value(self.deepface_retinaface, settings.get("deepface_use_retinaface", True))
        self._set_checkbox_value(self.fer_enabled, settings.get("fer_enabled", False))

        self._set_llm_value(settings.get("llm_model", self.llm_options[0]["value"]))

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

    def _set_llm_value(self, value):
        value_to_label = {opt["value"]: opt["label"] for opt in self.llm_options}
        label = value_to_label.get(value, self.llm_options[0]["label"])
        self.llm_combo.blockSignals(True)
        self.llm_combo.setCurrentText(label)
        self.llm_combo.blockSignals(False)

    def update_geometry(self, parent_size):
        max_w = int(parent_size.width() * 0.6)
        max_h = int(parent_size.height() * 0.85)
        min_w = 420
        min_h = 420
        menu_w = max(min_w, min(max_w, parent_size.width() - 40))
        menu_h = max(min_h, min(max_h, parent_size.height() - 40))
        self.resize(menu_w, menu_h)
        self.move((parent_size.width() - menu_w) // 2, (parent_size.height() - menu_h) // 2)

