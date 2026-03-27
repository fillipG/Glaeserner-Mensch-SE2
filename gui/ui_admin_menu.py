"""
Name: "ui_admin_menu.py"
Beschreibung: Stellt das Admin-Menue mit Laufzeitkonfiguration fuer die GUI bereit.
Autor: Fillip Giffhorn und Florian Höft
"""

from PyQt6.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QSlider, QCheckBox, QLineEdit, QScrollArea, QSizePolicy, QComboBox,
                             QRadioButton)
from PyQt6.QtGui import QFont, QIntValidator
from PyQt6.QtCore import Qt, pyqtSignal


class AdminMenu(QFrame):
    """Admin-Menue mit Anzeige- und Slider-Elementen."""
    photo_delay_changed = pyqtSignal(int)
    close_on_no_person_enabled_changed = pyqtSignal(bool)
    close_on_no_person_changed = pyqtSignal(int)
    animation_speed_changed = pyqtSignal(int)
    pipeline_timeout_changed = pyqtSignal(int)
    face_yolo_confidence_changed = pyqtSignal(float)
    body_yolo_confidence_changed = pyqtSignal(float)
    body_padding_ratio_changed = pyqtSignal(float)
    fullscreen_toggled = pyqtSignal(bool)
    developer_mode_toggled = pyqtSignal(bool)
    pool_enabled_changed = pyqtSignal(bool)
    pool_max_extra_changed = pyqtSignal(int)
    pool_cooldown_changed = pyqtSignal(int)
    moondream_enabled_changed = pyqtSignal(bool)
    moondream_prompt_changed = pyqtSignal(str)
    moondream_crop_mode_changed = pyqtSignal(str)
    ollama_enabled_changed = pyqtSignal(bool)
    ollama_prompt_changed = pyqtSignal(str)
    deepface_enabled_changed = pyqtSignal(bool)
    deepface_retinaface_changed = pyqtSignal(bool)
    fer_enabled_changed = pyqtSignal(bool)
    live_deepface_enabled_changed = pyqtSignal(bool)
    live_deepface_interval_changed = pyqtSignal(int)
    sounds_enabled_changed = pyqtSignal(bool)
    llm_model_changed = pyqtSignal(str)
    reset_defaults_requested = pyqtSignal()
    statistics_enabled_changed = pyqtSignal(bool)
    statistics_reset_requested = pyqtSignal()

    def __init__(self, llm_options=None, parent=None):
        """
        Initialisiert das Admin-Menue inklusive aller Controls.
        :param llm_options: Verfuegbare LLM-Auswahloptionen.
        :param parent: Optionales Parent-Widget.
        """
        super().__init__(parent)
        self.llm_options = llm_options or []
        self._face_yolo_min = 0.10
        self._face_yolo_max = 0.90
        self._face_yolo_step = 0.05
        self._body_padding_min = 0.00
        self._body_padding_max = 0.50
        self._body_padding_step = 0.01
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

        self.photo_delay_label = QLabel("Wartezeit (Sek.)")
        self.photo_delay_value = QLabel("1 s")
        self.photo_delay_slider = self._create_slider(1, 60)
        self.photo_delay_slider.valueChanged.connect(self._on_photo_delay_changed)
        general_layout.addLayout(
            self._slider_row(self.photo_delay_label, self.photo_delay_slider, self.photo_delay_value)
        )

        self.close_on_no_person_button = QPushButton("Auto-Close: AN")
        self.close_on_no_person_button.setCheckable(True)
        self.close_on_no_person_button.setStyleSheet(
            "QPushButton { background-color: #3d2b1f; color: #f4e4bc; border: 2px solid #f4e4bc; "
            "border-radius: 10px; padding: 8px 14px; font-size: 16px; font-weight: bold; }"
            "QPushButton:checked { background-color: #5a4030; }"
        )
        self.close_on_no_person_button.toggled.connect(self._on_close_on_no_person_enabled_toggled)
        general_layout.addWidget(self.close_on_no_person_button)

        self.close_on_no_person_label = QLabel("Auto-Close ohne Person")
        self.close_on_no_person_value = QLabel("10 s")
        self.close_on_no_person_slider = self._create_slider(5, 60)
        self.close_on_no_person_slider.setSingleStep(5)
        self.close_on_no_person_slider.setPageStep(5)
        self.close_on_no_person_slider.valueChanged.connect(self._on_close_on_no_person_changed)
        general_layout.addLayout(
            self._slider_row(
                self.close_on_no_person_label,
                self.close_on_no_person_slider,
                self.close_on_no_person_value
            )
        )

        self.anim_speed_label = QLabel("Animationsgeschwindigkeit")
        self.anim_speed_value = QLabel("10")
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

        self.face_yolo_confidence_label = QLabel("Face-YOLO Confidence")
        self.face_yolo_confidence_value = QLabel("0.50")
        self.face_yolo_confidence_slider = self._create_slider(10, 90)
        self.face_yolo_confidence_slider.setSingleStep(5)
        self.face_yolo_confidence_slider.setPageStep(5)
        self.face_yolo_confidence_slider.valueChanged.connect(self._on_face_yolo_confidence_changed)
        general_layout.addLayout(
            self._slider_row(
                self.face_yolo_confidence_label,
                self.face_yolo_confidence_slider,
                self.face_yolo_confidence_value
            )
        )

        face_yolo_hint = QLabel(
            "Niedriger = mehr Gesichter erkannt, aber ungenauer\n"
            "Hoeher = weniger Erkennungen, aber zuverlaessiger\n"
            "Empfohlen: 0.50"
        )
        face_yolo_hint.setStyleSheet("color: rgba(244, 228, 188, 150); font-size: 12px;")
        face_yolo_hint.setWordWrap(True)
        general_layout.addWidget(face_yolo_hint)

        self.body_yolo_confidence_label = QLabel("Body-YOLO Confidence")
        self.body_yolo_confidence_value = QLabel("0.35")
        self.body_yolo_confidence_slider = self._create_slider(10, 90)
        self.body_yolo_confidence_slider.setSingleStep(5)
        self.body_yolo_confidence_slider.setPageStep(5)
        self.body_yolo_confidence_slider.valueChanged.connect(self._on_body_yolo_confidence_changed)
        general_layout.addLayout(
            self._slider_row(
                self.body_yolo_confidence_label,
                self.body_yolo_confidence_slider,
                self.body_yolo_confidence_value
            )
        )

        self.body_padding_label = QLabel("Body-Crop Padding")
        self.body_padding_value = QLabel("0.12")
        self.body_padding_slider = self._create_slider(0, 50)
        self.body_padding_slider.setSingleStep(1)
        self.body_padding_slider.setPageStep(5)
        self.body_padding_slider.valueChanged.connect(self._on_body_padding_ratio_changed)
        general_layout.addLayout(
            self._slider_row(
                self.body_padding_label,
                self.body_padding_slider,
                self.body_padding_value
            )
        )

        body_yolo_hint = QLabel(
            "Nur relevant fuer Body-/Seg-Crops\n"
            "Confidence steuert Personenerkennung, Padding vergroessert den Koerper-Crop"
        )
        body_yolo_hint.setStyleSheet("color: rgba(244, 228, 188, 150); font-size: 12px;")
        body_yolo_hint.setWordWrap(True)
        general_layout.addWidget(body_yolo_hint)

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

        self.sounds_enabled = QCheckBox("Sounds aktiviert")
        self.sounds_enabled.setStyleSheet("QCheckBox { font-size: 14px; }")
        self.sounds_enabled.toggled.connect(self.sounds_enabled_changed)
        graphics_layout.addWidget(self.sounds_enabled)
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

        self.pool_cooldown_label = QLabel("Cooldown (Durchlaeufe)")
        self.pool_cooldown_value = QLabel("0")
        self.pool_cooldown_slider = self._create_slider(0, 10)
        self.pool_cooldown_slider.valueChanged.connect(self._on_pool_cooldown_changed)
        pool_layout.addLayout(
            self._slider_row(self.pool_cooldown_label, self.pool_cooldown_slider, self.pool_cooldown_value)
        )
        layout.addWidget(pool_box)

        layout.addWidget(self._section_title("KI-MODELLE"))
        models_box = self._create_group_box()
        models_layout = QVBoxLayout(models_box)

        self.moondream_enabled, self.moondream_prompt, moondream_layout = self._create_model_block(
            models_layout, "Moondream", has_prompt=True
        )
        self.moondream_enabled.toggled.connect(self.moondream_enabled_changed)
        self.moondream_enabled.setToolTip("Moondream ist fest aktiviert und kann nicht ausgeschaltet werden.")
        self.moondream_enabled.setStyleSheet(
            "QCheckBox { font-size: 14px; }"
            "QCheckBox:disabled { color: rgba(244, 228, 188, 120); }"
        )
        self.moondream_prompt.editingFinished.connect(self._on_moondream_prompt_changed)

        # Moondream kann im Admin-Menue zwischen Face-, Body- und Segmentation-Crop umgeschaltet werden.
        # Der Entwicklermodus "shadow" bleibt absichtlich nur in der config.yaml sichtbar.
        crop_mode_label = QLabel("Moondream Bildquelle")
        moondream_layout.addWidget(crop_mode_label)
        self.moondream_crop_face = QRadioButton("Gesicht (Face-Crop)")
        self.moondream_crop_body = QRadioButton("Koerper (YOLO Box-Crop)")
        self.moondream_crop_body_seg = QRadioButton("Koerper (YOLO-Seg freigestellt)")
        self.moondream_crop_face.setStyleSheet("QRadioButton { font-size: 14px; }")
        self.moondream_crop_body.setStyleSheet("QRadioButton { font-size: 14px; }")
        self.moondream_crop_body_seg.setStyleSheet("QRadioButton { font-size: 14px; }")
        self.moondream_crop_face.setToolTip(
            "Face-YOLO liefert einen klassischen Gesichts-Crop fuer Moondream."
        )
        self.moondream_crop_body.setToolTip(
            "YOLO Person-Detection liefert einen rechteckigen Koerper-Crop inklusive Rest-Hintergrund."
        )
        self.moondream_crop_body_seg.setToolTip(
            "YOLO Segmentation stellt die Person pixelgenau frei und setzt den Hintergrund weiss."
        )
        self.moondream_crop_face.toggled.connect(
            lambda checked: self.moondream_crop_mode_changed.emit("face") if checked else None
        )
        self.moondream_crop_body.toggled.connect(
            lambda checked: self.moondream_crop_mode_changed.emit("body") if checked else None
        )
        self.moondream_crop_body_seg.toggled.connect(
            lambda checked: self.moondream_crop_mode_changed.emit("body_seg") if checked else None
        )
        moondream_layout.addWidget(self.moondream_crop_face)
        moondream_layout.addWidget(self.moondream_crop_body)
        moondream_layout.addWidget(self.moondream_crop_body_seg)

        self.ollama_enabled, self.ollama_prompt, _ = self._create_model_block(
            models_layout, "Ollama", has_prompt=True
        )
        self.ollama_enabled.toggled.connect(self.ollama_enabled_changed)
        self.ollama_prompt.editingFinished.connect(self._on_ollama_prompt_changed)

        self.deepface_enabled, _, deepface_layout = self._create_model_block(models_layout, "Deepface")
        self.deepface_enabled.toggled.connect(self.deepface_enabled_changed)
        self.deepface_retinaface = QCheckBox("Verbessertes Analysemodell (RetinaFace)")
        self.deepface_retinaface.setStyleSheet("QCheckBox { font-size: 14px; }")
        self.deepface_retinaface.toggled.connect(self.deepface_retinaface_changed)
        deepface_layout.addWidget(self.deepface_retinaface)

        self.fer_enabled, _, _ = self._create_model_block(models_layout, "FER")
        self.fer_enabled.toggled.connect(self.fer_enabled_changed)

        layout.addWidget(models_box)

        layout.addWidget(self._section_title("LIVE-ANALYSE"))
        live_analysis_box = self._create_group_box()
        live_analysis_layout = QVBoxLayout(live_analysis_box)
        self.live_deepface_enabled = QCheckBox("Live-Analyse (Alter / Geschlecht / Emotion)")
        self.live_deepface_enabled.setStyleSheet("QCheckBox { font-size: 14px; }")
        self.live_deepface_enabled.toggled.connect(self.live_deepface_enabled_changed)
        live_analysis_layout.addWidget(self.live_deepface_enabled)
        live_interval_row = QHBoxLayout()
        live_interval_label = QLabel("Intervall (Sek.)")
        live_interval_label.setMinimumWidth(220)
        self.live_deepface_interval_edit = QLineEdit()
        self.live_deepface_interval_edit.setValidator(QIntValidator(1, 60, self))
        self.live_deepface_interval_edit.setPlaceholderText("z.B. 3")
        self.live_deepface_interval_edit.setStyleSheet(
            "QLineEdit { background-color: #2b2018; color: #f4e4bc; border: 1px solid #f4e4bc; "
            "border-radius: 6px; padding: 6px; }"
        )
        self.live_deepface_interval_edit.editingFinished.connect(self._on_live_deepface_interval_changed)
        live_interval_row.addWidget(live_interval_label)
        live_interval_row.addWidget(self.live_deepface_interval_edit, 1)
        live_analysis_layout.addLayout(live_interval_row)
        layout.addWidget(live_analysis_box)

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

        layout.addWidget(self._section_title("BESUCHERSTATISTIK"))
        stats_box = self._create_group_box()
        stats_layout = QVBoxLayout(stats_box)

        self.stats_enabled_button = QPushButton("Statistik: AN")
        self.stats_enabled_button.setCheckable(True)
        self.stats_enabled_button.setChecked(True)
        self.stats_enabled_button.setStyleSheet(
            "QPushButton { background-color: #3d5a3e; color: #f4e4bc; border: 1px solid #f4e4bc; "
            "border-radius: 6px; padding: 6px 12px; }"
            "QPushButton:checked { background-color: #3d5a3e; }"
            "QPushButton:!checked { background-color: #5a3d3e; }"
        )
        self.stats_enabled_button.toggled.connect(self._on_stats_enabled_toggled)
        stats_layout.addWidget(self.stats_enabled_button)

        self.stats_reset_button = QPushButton("Statistik zurücksetzen")
        self.stats_reset_button.setStyleSheet(
            "QPushButton { background-color: #5a3d2b; color: #f4e4bc; border: 1px solid #f4e4bc; "
            "border-radius: 6px; padding: 6px 12px; }"
            "QPushButton:pressed { background-color: #7a3d2b; }"
        )
        self.stats_reset_button.clicked.connect(self._on_stats_reset_clicked)
        stats_layout.addWidget(self.stats_reset_button)

        # Read-only Anzeige der Statistik (alle Zeiträume immer sichtbar)
        self.stats_today_label = QLabel("Heute: – Durchgänge, – Personen")
        self.stats_today_label.setFont(QFont("Graduate", 9))
        self.stats_today_label.setStyleSheet("color: #c8b89a;")
        stats_layout.addWidget(self.stats_today_label)

        self.stats_period_label = QLabel("Letzte 7 Tage: – Durchgänge, – Personen")
        self.stats_period_label.setFont(QFont("Graduate", 9))
        self.stats_period_label.setStyleSheet("color: #c8b89a;")
        stats_layout.addWidget(self.stats_period_label)

        self.stats_month_label = QLabel("Dieser Monat: – Durchgänge, – Personen")
        self.stats_month_label.setFont(QFont("Graduate", 9))
        self.stats_month_label.setStyleSheet("color: #c8b89a;")
        stats_layout.addWidget(self.stats_month_label)

        self.stats_total_label = QLabel("Gesamt: – Durchgänge, – Personen")
        self.stats_total_label.setFont(QFont("Graduate", 9))
        self.stats_total_label.setStyleSheet("color: #c8b89a;")
        stats_layout.addWidget(self.stats_total_label)

        layout.addWidget(stats_box)

        self.reset_defaults_button = QPushButton("AUF STANDARDEINSTELLUNGEN ZURUECKSETZEN")
        self.reset_defaults_button.setStyleSheet(
            "QPushButton { background-color: #6b2e1f; color: #f4e4bc; border: 2px solid #f4e4bc; "
            "border-radius: 10px; padding: 10px 14px; font-size: 15px; font-weight: bold; }"
            "QPushButton:hover { background-color: #83402f; }"
            "QPushButton:pressed { background-color: #4a1f14; }"
        )
        self.reset_defaults_button.clicked.connect(self.reset_defaults_requested.emit)
        layout.addWidget(self.reset_defaults_button)

        layout.addStretch()
        layout.addWidget(QLabel("DRUECKE 'E' ZUM VERLASSEN", alignment=Qt.AlignmentFlag.AlignCenter))
        self.hide()

    def _section_title(self, text):
        """
        Erstellt einen Abschnittstitel fuer das Menue.
        :param text: Titeltext.
        :return: Fertig konfiguriertes QLabel.
        """
        label = QLabel(text)
        label.setFont(QFont("Graduate", 16, QFont.Weight.Bold))
        label.setStyleSheet("color: #f4e4bc;")
        return label

    def _create_group_box(self):
        """
        Erstellt einen optisch einheitlichen Gruppencontainer.
        :return: Konfigurierter QFrame.
        """
        box = QFrame()
        box.setStyleSheet(
            "QFrame { background-color: rgba(61, 43, 31, 200); border: 1px solid rgba(244, 228, 188, 120); "
            "border-radius: 12px; }"
        )
        box.setContentsMargins(10, 8, 10, 8)
        return box

    def _create_slider(self, min_val, max_val):
        """
        Erstellt einen horizontalen Slider mit Standardstil.
        :param min_val: Minimalwert.
        :param max_val: Maximalwert.
        :return: Konfigurierter Slider.
        """
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(min_val, max_val)
        s.setStyleSheet(
            "QSlider::handle:horizontal { background: #f4e4bc; width: 18px; border-radius: 9px; } "
            "QSlider::groove:horizontal { background: #3d2b1f; height: 10px; border-radius: 5px; }"
        )
        return s

    def _slider_row(self, label, slider, value_label):
        """
        Baut eine Zeile aus Label, Slider und Wertanzeige.
        :param label: Beschriftungslabel.
        :param slider: Slider-Widget.
        :param value_label: Label fuer den numerischen Wert.
        :return: Layoutzeile fuer die UI.
        """
        row = QHBoxLayout()
        label.setMinimumWidth(220)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        value_label.setMinimumWidth(60)
        row.addWidget(label)
        row.addWidget(slider, 1)
        row.addWidget(value_label)
        return row

    def _create_model_block(self, parent_layout, title, has_prompt=False):
        """
        Erstellt den UI-Block fuer ein KI-Modell.
        :param parent_layout: Ziel-Layout fuer den Block.
        :param title: Anzeigename des Modells.
        :param has_prompt: True, wenn ein Prompt-Feld benoetigt wird.
        :return: Tuple aus Aktiv-Checkbox, Prompt-Input und Block-Layout.
        """
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

    def _on_photo_delay_changed(self, value):
        """
        Reagiert auf Aenderungen der Fotoverzoegerung.
        :param value: Neuer Wert in Sekunden.
        """
        self.photo_delay_value.setText(f"{value} s")
        self.photo_delay_changed.emit(value)

    def _on_close_on_no_person_changed(self, value):
        """
        Rundet Auto-Close-Werte auf 5er-Schritte und emittiert sie.
        :param value: Neuer Sliderwert.
        """
        rounded_value = max(5, min(60, int(round(value / 5.0) * 5)))
        if rounded_value != value:
            self._set_slider_value(self.close_on_no_person_slider, rounded_value)
        self.close_on_no_person_value.setText(f"{rounded_value} s")
        self.close_on_no_person_changed.emit(rounded_value)

    def _on_close_on_no_person_enabled_toggled(self, checked):
        """
        Schaltet den Auto-Close-Bereich ein oder aus.
        :param checked: True aktiviert Auto-Close.
        """
        self.close_on_no_person_button.setText("Auto-Close: AN" if checked else "Auto-Close: AUS")
        self.close_on_no_person_label.setEnabled(checked)
        self.close_on_no_person_slider.setEnabled(checked)
        self.close_on_no_person_value.setEnabled(checked)
        self.close_on_no_person_enabled_changed.emit(checked)

    def _on_animation_speed_changed(self, value):
        """
        Reagiert auf Aenderungen der Animationsgeschwindigkeit.
        :param value: Neuer Geschwindigkeitswert.
        """
        self.anim_speed_value.setText(str(value))
        self.animation_speed_changed.emit(value)

    def _on_pipeline_timeout_changed(self):
        """
        Validiert und emittiert den Pipeline-Timeout aus dem Eingabefeld.
        """
        text = self.pipeline_timeout_edit.text().strip()
        if not text:
            return
        try:
            value = max(1, int(text))
        except ValueError:
            return
        self.pipeline_timeout_edit.setText(str(value))
        self.pipeline_timeout_changed.emit(value)

    def _on_face_yolo_confidence_changed(self, slider_value):
        """
        Reagiert auf Aenderungen der Face-YOLO-Confidence.
        :param slider_value: Sliderwert zwischen 10 und 90.
        """
        value = slider_value / 100.0
        self.face_yolo_confidence_value.setText(f"{value:.2f}")
        self.face_yolo_confidence_changed.emit(value)

    def _on_body_yolo_confidence_changed(self, slider_value):
        """
        Reagiert auf Aenderungen der Body-YOLO-Confidence.
        :param slider_value: Sliderwert zwischen 10 und 90.
        """
        value = slider_value / 100.0
        self.body_yolo_confidence_value.setText(f"{value:.2f}")
        self.body_yolo_confidence_changed.emit(value)

    def _on_body_padding_ratio_changed(self, slider_value):
        """
        Reagiert auf Aenderungen des Body-Crop-Paddings.
        :param slider_value: Sliderwert zwischen 0 und 50.
        """
        value = slider_value / 100.0
        self.body_padding_value.setText(f"{value:.2f}")
        self.body_padding_ratio_changed.emit(value)

    def _on_fullscreen_toggled(self, checked):
        """
        Reagiert auf Vollbild-Umschaltung.
        :param checked: True aktiviert Vollbild.
        """
        self.fullscreen_button.setText("Vollbild: AN" if checked else "Vollbild: AUS")
        self.fullscreen_toggled.emit(checked)

    def _on_developer_mode_toggled(self, checked):
        """
        Reagiert auf Developer-Mode-Umschaltung.
        :param checked: True aktiviert Developer Mode.
        """
        self.developer_mode_button.setText("Developer Mode: AN" if checked else "Developer Mode: AUS")
        self.developer_mode_toggled.emit(checked)

    def _on_pool_enabled_toggled(self, checked):
        """
        Reagiert auf Pool-Umschaltung.
        :param checked: True aktiviert den Pool.
        """
        self.pool_enabled_button.setText("Pool: AN" if checked else "Pool: AUS")
        self.pool_enabled_changed.emit(checked)

    def _on_pool_max_extra_changed(self, value):
        """
        Reagiert auf Aenderung der Zusatzpersonen im Pool.
        :param value: Neue Anzahl Zusatzpersonen.
        """
        self.pool_max_extra_value.setText(str(value))
        self.pool_max_extra_changed.emit(value)

    def _on_pool_cooldown_changed(self, value):
        """
        Reagiert auf Aenderung des Pool-Cooldowns.
        :param value: Neue Cooldown-Anzahl.
        """
        self.pool_cooldown_value.setText(str(value))
        self.pool_cooldown_changed.emit(value)

    def _on_moondream_prompt_changed(self):
        """
        Emittiert den aktuellen Moondream-Prompt.
        """
        if self.moondream_prompt is None:
            return
        self.moondream_prompt_changed.emit(self.moondream_prompt.text().strip())

    def _on_ollama_prompt_changed(self):
        """
        Emittiert den aktuellen Ollama-Prompt.
        """
        if self.ollama_prompt is None:
            return
        self.ollama_prompt_changed.emit(self.ollama_prompt.text().strip())

    def _on_llm_changed(self, text):
        """
        Mapped die sichtbare LLM-Auswahl auf den internen Modellwert.
        :param text: Angezeigtes Label aus der ComboBox.
        """
        label_to_value = {opt["label"]: opt["value"] for opt in self.llm_options}
        value = label_to_value.get(text, self.llm_options[0]["value"])
        self.llm_model_changed.emit(value)

    def _on_live_deepface_interval_changed(self):
        """
        Validiert und emittiert das Live-DeepFace-Intervall aus dem Eingabefeld.
        """
        text = self.live_deepface_interval_edit.text().strip()
        if not text:
            return
        try:
            value = max(1, int(text))
        except ValueError:
            return
        self.live_deepface_interval_edit.setText(str(value))
        self.live_deepface_interval_changed.emit(value)

    def _on_stats_enabled_toggled(self, checked):
        """
        Schaltet die Besucherstatistik ein oder aus.
        :param checked: True aktiviert die Statistik.
        """
        self.stats_enabled_button.setText("Statistik: AN" if checked else "Statistik: AUS")
        self.statistics_enabled_changed.emit(checked)

    def _on_stats_reset_clicked(self):
        """
        Emittiert das Reset-Signal nach Bestätigung durch den Admin.
        """
        self.statistics_reset_requested.emit()

    def refresh_stats(self, today_data, week_data, month_data, total_data):
        """
        Aktualisiert alle vier Statistik-Labels im Admin-Menü.
        Wird beim Öffnen des Menüs und nach jeder Änderung aufgerufen.
        :param today_data: Dict mit 'durchgaenge' und 'personen' für heute.
        :param week_data: Dict mit 'durchgaenge' und 'personen' der letzten 7 Tage.
        :param month_data: Dict mit 'durchgaenge' und 'personen' des aktuellen Monats.
        :param total_data: Dict mit 'durchgaenge' und 'personen' gesamt.
        """
        d = today_data or {"durchgaenge": 0, "personen": 0}
        self.stats_today_label.setText(
            f"Heute: {d['durchgaenge']} Durchgänge, {d['personen']} Personen"
        )
        w = week_data or {"durchgaenge": 0, "personen": 0}
        self.stats_period_label.setText(
            f"Letzte 7 Tage: {w['durchgaenge']} Durchgänge, {w['personen']} Personen"
        )
        m = month_data or {"durchgaenge": 0, "personen": 0}
        self.stats_month_label.setText(
            f"Dieser Monat: {m['durchgaenge']} Durchgänge, {m['personen']} Personen"
        )
        t = total_data or {"durchgaenge": 0, "personen": 0}
        self.stats_total_label.setText(
            f"Gesamt: {t['durchgaenge']} Durchgänge, {t['personen']} Personen"
        )

    def apply_settings(self, settings):
        """
        Synchronisiert alle Menueelemente mit den uebergebenen Einstellungen.
        :param settings: Dictionary mit Admin-Werten.
        """
        self._set_slider_value(self.photo_delay_slider, settings.get("photo_delay", 3))
        self.photo_delay_value.setText(f"{self.photo_delay_slider.value()} s")
        self._set_toggle_button(
            self.close_on_no_person_button,
            settings.get("close_on_no_person_enabled", True)
        )
        self.close_on_no_person_button.setText(
            "Auto-Close: AN" if self.close_on_no_person_button.isChecked() else "Auto-Close: AUS"
        )
        self._set_slider_value(
            self.close_on_no_person_slider,
            max(5, min(60, int(settings.get("close_on_no_person_seconds", 10))))
        )
        self.close_on_no_person_value.setText(f"{self.close_on_no_person_slider.value()} s")
        self.close_on_no_person_label.setEnabled(self.close_on_no_person_button.isChecked())
        self.close_on_no_person_slider.setEnabled(self.close_on_no_person_button.isChecked())
        self.close_on_no_person_value.setEnabled(self.close_on_no_person_button.isChecked())
        self._set_slider_value(self.anim_speed_slider, settings.get("animation_speed", 10))
        self.anim_speed_value.setText(str(self.anim_speed_slider.value()))
        self._set_lineedit_value(self.pipeline_timeout_edit, str(settings.get("pipeline_timeout_seconds", 120)))
        self._set_slider_value(
            self.face_yolo_confidence_slider,
            self._face_yolo_to_slider_value(float(settings.get("face_yolo_confidence", 0.5)))
        )
        self.face_yolo_confidence_value.setText(
            f"{self.face_yolo_confidence_slider.value() / 100.0:.2f}"
        )
        self._set_slider_value(
            self.body_yolo_confidence_slider,
            self._face_yolo_to_slider_value(float(settings.get("body_yolo_confidence", 0.35)))
        )
        self.body_yolo_confidence_value.setText(
            f"{self.body_yolo_confidence_slider.value() / 100.0:.2f}"
        )
        self._set_slider_value(
            self.body_padding_slider,
            self._body_padding_to_slider_value(float(settings.get("body_padding_ratio", 0.12)))
        )
        self.body_padding_value.setText(
            f"{self.body_padding_slider.value() / 100.0:.2f}"
        )

        self._set_toggle_button(self.fullscreen_button, settings.get("fullscreen", True))
        self.fullscreen_button.setText("Vollbild: AN" if self.fullscreen_button.isChecked() else "Vollbild: AUS")

        self._set_toggle_button(self.developer_mode_button, settings.get("developer_mode", False))
        self.developer_mode_button.setText(
            "Developer Mode: AN" if self.developer_mode_button.isChecked() else "Developer Mode: AUS"
        )
        self._set_checkbox_value(
            self.sounds_enabled,
            settings.get("sounds_enabled", True)
        )

        self._set_toggle_button(self.pool_enabled_button, settings.get("pool_enabled", True))
        self.pool_enabled_button.setText("Pool: AN" if self.pool_enabled_button.isChecked() else "Pool: AUS")
        self._set_slider_value(self.pool_max_extra_slider, settings.get("pool_max_extra_persons", 3))
        self.pool_max_extra_value.setText(str(self.pool_max_extra_slider.value()))
        self._set_slider_value(self.pool_cooldown_slider, settings.get("pool_cooldown_batches", 3))
        self.pool_cooldown_value.setText(str(self.pool_cooldown_slider.value()))

        self.moondream_enabled.blockSignals(True)
        self.moondream_enabled.setChecked(True)
        self.moondream_enabled.blockSignals(False)
        self.moondream_enabled.setEnabled(False)
        if self.moondream_prompt is not None:
            self._set_lineedit_value(self.moondream_prompt, settings.get("moondream_prompt", ""))
        self._set_moondream_crop_mode(settings.get("moondream_crop_mode", "face"))
        self._set_checkbox_value(self.ollama_enabled, settings.get("ollama_enabled", True))
        if self.ollama_prompt is not None:
            self._set_lineedit_value(self.ollama_prompt, settings.get("ollama_prompt", ""))
        self._set_checkbox_value(self.deepface_enabled, settings.get("deepface_enabled", False))
        self._set_checkbox_value(self.deepface_retinaface, settings.get("deepface_use_retinaface", True))
        self._set_checkbox_value(self.fer_enabled, settings.get("fer_enabled", False))
        self._set_checkbox_value(self.live_deepface_enabled, settings.get("live_deepface_enabled", False))
        self._set_lineedit_value(
            self.live_deepface_interval_edit,
            str(settings.get("live_deepface_interval_seconds", 3))
        )

        self._set_llm_value(settings.get("llm_model", self.llm_options[0]["value"]))

        stats_enabled = settings.get("statistics_enabled", True)
        self._set_toggle_button(self.stats_enabled_button, stats_enabled)
        self.stats_enabled_button.setText("Statistik: AN" if stats_enabled else "Statistik: AUS")

    def _set_slider_value(self, slider, value):
        """
        Setzt einen Sliderwert ohne Signale auszufeuern.
        :param slider: Ziel-Slider.
        :param value: Neuer Sliderwert.
        """
        slider.blockSignals(True)
        slider.setValue(int(value))
        slider.blockSignals(False)

    def _face_yolo_to_slider_value(self, value):
        """
        Quantisiert eine Float-Confidence auf den Sliderbereich.
        :param value: Face-YOLO-Confidence als Float.
        :return: Passender Integer-Sliderwert.
        """
        clamped = max(self._face_yolo_min, min(self._face_yolo_max, float(value)))
        step_index = round((clamped - self._face_yolo_min) / self._face_yolo_step)
        return int(round((self._face_yolo_min + step_index * self._face_yolo_step) * 100))

    def _body_padding_to_slider_value(self, value):
        """
        Quantisiert ein Body-Padding-Ratio auf den Sliderbereich.
        :param value: Body-Crop-Padding als Float.
        :return: Passender Integer-Sliderwert.
        """
        clamped = max(self._body_padding_min, min(self._body_padding_max, float(value)))
        step_index = round((clamped - self._body_padding_min) / self._body_padding_step)
        return int(round((self._body_padding_min + step_index * self._body_padding_step) * 100))

    def _set_toggle_button(self, button, checked):
        """
        Setzt den Zustand eines Toggle-Buttons signalfrei.
        :param button: Ziel-Button.
        :param checked: Neuer Togglezustand.
        """
        button.blockSignals(True)
        button.setChecked(bool(checked))
        button.blockSignals(False)

    def _set_checkbox_value(self, checkbox, checked):
        """
        Setzt den Zustand einer Checkbox signalfrei.
        :param checkbox: Ziel-Checkbox.
        :param checked: Neuer Checkboxzustand.
        """
        checkbox.blockSignals(True)
        checkbox.setChecked(bool(checked))
        checkbox.blockSignals(False)

    def _set_moondream_crop_mode(self, mode):
        """
        Setzt die sichtbare Moondream-Bildquelle ohne Signale.
        Shadow bleibt absichtlich unsichtbar und setzt die Radio-Buttons zurueck.
        :param mode: "face", "body", "body_seg" oder ein interner Entwicklermodus wie "shadow".
        """
        self.moondream_crop_face.blockSignals(True)
        self.moondream_crop_body.blockSignals(True)
        self.moondream_crop_body_seg.blockSignals(True)
        self.moondream_crop_face.setAutoExclusive(False)
        self.moondream_crop_body.setAutoExclusive(False)
        self.moondream_crop_body_seg.setAutoExclusive(False)
        self.moondream_crop_face.setChecked(mode == "face")
        self.moondream_crop_body.setChecked(mode == "body")
        self.moondream_crop_body_seg.setChecked(mode == "body_seg")
        if mode not in {"face", "body", "body_seg"}:
            self.moondream_crop_face.setChecked(False)
            self.moondream_crop_body.setChecked(False)
            self.moondream_crop_body_seg.setChecked(False)
        self.moondream_crop_face.setAutoExclusive(True)
        self.moondream_crop_body.setAutoExclusive(True)
        self.moondream_crop_body_seg.setAutoExclusive(True)
        self.moondream_crop_face.blockSignals(False)
        self.moondream_crop_body.blockSignals(False)
        self.moondream_crop_body_seg.blockSignals(False)

    def _set_lineedit_value(self, lineedit, text):
        """
        Setzt den Text eines Eingabefeldes signalfrei.
        :param lineedit: Ziel-Eingabefeld.
        :param text: Neuer Text.
        """
        lineedit.blockSignals(True)
        lineedit.setText(text)
        lineedit.blockSignals(False)

    def _set_llm_value(self, value):
        """
        Setzt die LLM-Auswahl ueber den internen Modellwert.
        :param value: Interner LLM-Wert.
        """
        value_to_label = {opt["value"]: opt["label"] for opt in self.llm_options}
        label = value_to_label.get(value, self.llm_options[0]["label"])
        self.llm_combo.blockSignals(True)
        self.llm_combo.setCurrentText(label)
        self.llm_combo.blockSignals(False)

    def update_geometry(self, parent_size):
        """
        Passt Groesse und Position des Menues an die Fenstergroesse an.
        :param parent_size: Verfuegbare Groesse des Parent-Fensters.
        """
        max_w = int(parent_size.width() * 0.6)
        max_h = int(parent_size.height() * 0.85)
        min_w = 420
        min_h = 420
        menu_w = max(min_w, min(max_w, parent_size.width() - 40))
        menu_h = max(min_h, min(max_h, parent_size.height() - 40))
        self.resize(menu_w, menu_h)
        self.move((parent_size.width() - menu_w) // 2, (parent_size.height() - menu_h) // 2)
