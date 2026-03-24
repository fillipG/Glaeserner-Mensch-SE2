"""
gui/main_gui.py
---------------
Haupt-GUI der Museumsanwendung "Gläserner Mensch".

Diese Klasse ist das Herzstück der Anwendung. Sie zeigt die Ordner-Animationen,
die Kamera-Vorschau und die generierten Personen-Akten an.

Struktur (Mixin-Pattern):
    Die ScalingAkteGUI erbt von mehreren Mixin-Klassen, die thematisch
    zusammengehörige Methoden bündeln. Dies hält die einzelnen Dateien klein
    und übersichtlich, ohne die Klassen-Struktur zu zerstückeln.

    Mixins (in gui/mixins/):
        AnimationMixin       – Ordner-Videos, Öffnen/Schließen
        CameraMixin          – Kamera-Preview, Gesichtserkennung
        PresenceMixin        – Auto-Close, Anwesenheits-Timer
        LanguageMixin        – Sprachwechsel, Button-Cooldown
        ResetMixin           – Reset-Countdown
        ConfigHandlersMixin  – Admin-Menü-Signale, Config I/O

    Diese Datei enthält:
        - __init__ (Initialisierung aller Zustandsvariablen und Timer)
        - Lade-Anzeige (LoadingSpinner)
        - Personen-Anzeige (PersonContainer, Sketch)
        - Buttons (Sprache, Reset)
        - Fenster-Events (keyPress, resize)

AUTOREN: Fillip Giffhorn, Lukas Käuper (Kamera + Logos), Florian Hoeft (Bug-Fixes)
"""

import os
import cv2

from PyQt6.QtWidgets import (QGraphicsView, QGraphicsScene, QFrame, QMessageBox)
from PyQt6.QtGui import QPixmap, QFont, QColor, QPainter, QImage, QPen
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, pyqtSlot

from sketch import create_advanced_sketch
from service import TranslationService
from config_service import ConfigService
from statistics_service import StatisticsService
from pool_loader import PoolLoader
from .description_repository import DescriptionRepository
from .gui_constants import SCENE_WIDTH, SCENE_HEIGHT, PATHS
from .gui_state import GUIState
from .live_deepface_service import LiveDeepFaceService
from .sound_service import SoundService
from .ui_admin_menu import AdminMenu
from .ui_person_container import PersonContainer
from .ui_widgets import AnimatedGraphicsButton, LoadingSpinnerItem

from .mixins.animation_mixin import AnimationMixin
from .mixins.camera_mixin import CameraMixin
from .mixins.presence_mixin import PresenceMixin
from .mixins.language_mixin import LanguageMixin
from .mixins.reset_mixin import ResetMixin
from .mixins.config_handlers_mixin import ConfigHandlersMixin


# Ollama-Modelle die im Admin-Menü zur Auswahl stehen
LLM_OPTIONS = [
    {"label": "Ollama - qwen2.5 3b (empfohlen)", "value": "qwen2.5:3b"},
    {"label": "Ollama - qwen3 4b (aktuell)", "value": "qwen3:4b"},
    {"label": "Ollama - gemma3 1b (aktuell, leicht)", "value": "gemma3:1b"},
    {"label": "Ollama - gemma3 4b (aktuell, stark)", "value": "gemma3:4b"},
    {"label": "Ollama - llama3.2 3b", "value": "llama3.2:3b"},
    {"label": "Ollama - llama3.2 1b (schnell)", "value": "llama3.2:1b"},
    {"label": "Ollama - gemma3", "value": "gemma3"},
    {"label": "Ollama - phi3 3.8b", "value": "phi3:3.8b"},
]


class ScalingAkteGUI(
    AnimationMixin,
    CameraMixin,
    PresenceMixin,
    LanguageMixin,
    ResetMixin,
    ConfigHandlersMixin,
    QGraphicsView,
):
    """
    Hauptfenster der Museumsanwendung.

    Koordiniert alle Teilbereiche über Mixin-Klassen und Qt-Signale.
    Läuft im Vollbild-Modus und ist für 24/7-Betrieb ausgelegt.

    Signale:
        folder_closed()                    → YOLOWorker: Kamera wieder starten
        presence_monitoring_requested()    → YOLOWorker: Anwesenheits-Modus
    """
    folder_closed                  = pyqtSignal()  # Akte geschlossen → YOLOWorker fortsetzen
    presence_monitoring_requested  = pyqtSignal()  # Akte offen → Anwesenheit überwachen

    def __init__(self):
        """
        Initialisiert alle Zustandsvariablen, Timer, Services und UI-Grundstruktur.
        Die eigentliche Anzeige wird durch show_closed_folder() am Ende aufgebaut.
        """
        super().__init__()
        self.scene = QGraphicsScene(0, 0, SCENE_WIDTH, SCENE_HEIGHT)
        self.setScene(self.scene)

        # ── Animations-Zustand ──────────────────────────────────────────────
        self.video_cap = None           # OpenCV VideoCapture für Animations-Videos
        self.video_item = None          # QGraphicsPixmapItem für Video-Frame
        self.is_animating = False       # True während ein Video abgespielt wird
        self._is_open = False           # True wenn die Akte geöffnet ist
        self.state = GUIState.CLOSED
        self._pending_close = None      # Ausstehender close_folder()-Aufruf während Animation
        self._animation_end_callback = None

        # ── Personen-Daten ───────────────────────────────────────────────────
        self.person_data = []
        self.active_containers = []

        # ── Lade-Anzeige ─────────────────────────────────────────────────────
        self.loading_item = None
        self.loading_active = False

        # ── Anwesenheits-Überwachung ──────────────────────────────────────────
        self._last_person_present = True
        self._missed_presence_checks = 0
        self._auto_close_monitoring_enabled = False
        self._auto_close_monitoring_pending = False

        # ── Reset-Button Zustand ──────────────────────────────────────────────
        self._reset_button_warning_active = False
        self._reset_button_pixmap_path = PATHS["reset_button"]
        self._reset_button_empty_path = PATHS["reset_button_empty"]
        self._reset_button_original_pixmap = None
        self._reset_countdown_item = None

        # Verzeichnis-Bereinigung beim Schließen
        self._clear_pipeline_outputs_on_close = False

        # Entwickler-Button (nur im Developer-Mode sichtbar)
        self.btn_open = None

        # ── Sprache ───────────────────────────────────────────────────────────
        # Cooldown verhindert Mehrfachklicks auf den Sprach-Button
        self._language_button_cooldown_ms = 5_000
        self._language_button_cooldown_timer = QTimer(self)
        self._language_button_cooldown_timer.setSingleShot(True)
        self._language_button_cooldown_timer.timeout.connect(self._on_language_button_cooldown_timeout)

        # ── Services ─────────────────────────────────────────────────────────
        self.config_service = ConfigService(default_llm_value=LLM_OPTIONS[0]["value"])
        self.config = self._load_config()
        self._live_deepface_svc = LiveDeepFaceService(self.config)
        self._sound_svc = SoundService(self.config)
        self._apply_runtime_settings_from_config()

        self.current_language = self.config.get("language", "de")
        self.translator = TranslationService(target_lang=self.current_language)
        self.description_repo = DescriptionRepository(PATHS["final_dir"])
        stats_cfg = self.config.get("statistics", {})
        self._stats_svc = StatisticsService(
            stats_file="visitor_stats.yaml",
            enabled=bool(stats_cfg.get("enabled", True)),
            retention_days=int(stats_cfg.get("retention_days", 365)),
        )
        self._pool_loader = PoolLoader("config.yaml", config_data=self.config)

        # ── Timer ─────────────────────────────────────────────────────────────
        # Foto-Countdown (CircularTimerItem vor dem Öffnen der Mappe)
        self.wait_timer = QTimer(self)
        self.wait_timer.timeout.connect(self._update_wait_timer)
        self.wait_timer_item = None
        self._wait_start_time = None
        self._wait_duration_s = 0

        # Reset-Countdown (Zahl über dem Reset-Button)
        self._reset_countdown_timer = QTimer(self)
        self._reset_countdown_timer.timeout.connect(self._update_reset_countdown)
        self._reset_countdown_remaining = 0

        # "Keine Person"-Warn-Timer (lässt Reset-Button blinken)
        self._no_person_warning_timer = QTimer(self)
        self._no_person_warning_timer.timeout.connect(self._blink_no_person_warning)

        # Pipeline-Timeout (schließt Akte wenn KIs zu lange brauchen)
        self._pipeline_timeout_timer = QTimer(self)
        self._pipeline_timeout_timer.setSingleShot(True)
        self._pipeline_timeout_timer.timeout.connect(self._on_pipeline_timeout)

        # ── Kamera-Vorschau ───────────────────────────────────────────────────
        # Haar-Cascade für Bounding-Boxes im Live-Bild (Autor: Lukas Käuper)
        self._face_cascade = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )
        # Gesichtserkennung maximal alle 50ms ausführen (rate-limiting)
        self._face_detect_interval_ms = 50
        self._last_face_detect_ms = 0
        self._last_faces = []
        # Auf 50% skalieren für schnellere Cascade-Detection
        self._face_detection_scale = 0.5
        self.camera_pixmap_item = None
        self._last_camera_preview_pixmap = None

        # ── Admin-Menü ────────────────────────────────────────────────────────
        self.admin_menu = AdminMenu(LLM_OPTIONS, self)
        self._connect_admin_menu()
        self._sync_admin_menu_with_config()

        # ── Qt-Rendering ─────────────────────────────────────────────────────
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)

        # Initiale Anzeige: geschlossener Ordner
        self.show_closed_folder()

    # =========================================================
    # Daten-Normalisierung
    # =========================================================

    def _normalize_person_data(self, personen_daten):
        """
        Normalisiert Personendaten auf maximal 4 Einträge.
        Akzeptiert Listen und andere iterierbare Typen.
        :return: Liste mit maximal 4 Personen-Dicts.
        """
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

    # =========================================================
    # Zustandsverwaltung
    # =========================================================

    def _set_state(self, state):
        """
        Setzt den internen GUI-Zustand.
        :param state: Neuer GUIState-Wert.
        """
        self.state = state

    def _calculate_danger(self, emotion):
        """
        Leitet die Gefahrenstufe aus der Emotion ab.
        Muss mit der Pipeline-Logik konsistent bleiben, damit Datei- und GUI-Weg
        dieselbe Bewertung anzeigen.
        """
        mapping = {
            "happy": "GERING",
            "neutral": "MITTEL",
            "surprise": "MITTEL",
            "surprised": "MITTEL",
            "sad": "HOCH",
            "fear": "HOCH",
            "fearful": "HOCH",
            "disgust": "HOCH",
            "disgusted": "HOCH",
            "angry": "EXTREM",
        }
        return mapping.get(str(emotion).lower().strip(), "MITTEL")

    def _build_person_dict_from_final(self, face_index):
        """
        Baut ein GUI-Personenobjekt direkt aus den YAML-Dateien im final-Ordner.
        """
        deepface_data = self.description_repo.read_deepface_data(face_index - 1) or {}
        from constants import PipelineStage
        ollama_enabled = bool((self._get_pipeline_entry(PipelineStage.OLLAMA) or {}).get("enabled", False))
        if ollama_enabled:
            description = self.description_repo.read_ollama_description(face_index - 1)
        else:
            description = self.description_repo.read_moondream_description(face_index - 1)

        sketch_path = os.path.join(PATHS["sketch_dir"], f"face{face_index}.png")
        return {
            "titel": f"ID: FACE{face_index}",
            "geschlecht": deepface_data.get("Geschlecht", "Unbekannt"),
            "augen": "Braun",
            "stimmung": deepface_data.get("Emotion", "Neutral"),
            "alter": str(deepface_data.get("Alter", "N/A")),
            "gefahr": self._calculate_danger(deepface_data.get("Emotion", "Neutral")),
            "beschreibung": description or "Keine Beschreibung gefunden.",
            "face_image_path": sketch_path if os.path.exists(sketch_path) else None,
            "source": "real",
        }

    def _append_pool_people(self, personen_daten):
        """
        Fuellt echte Personen bei Bedarf mit Pool-Personen auf, damit 4 Akten
        angezeigt werden koennen.
        """
        real_count = len(personen_daten)
        if real_count == 0 or real_count >= 4:
            return personen_daten

        pool_selection = self._pool_loader.get_pool_persons(4 - real_count)
        for pool_person in pool_selection:
            deepface_data = pool_person.get("deepface", {})
            description_data = pool_person.get("ollama", {})
            personen_daten.append({
                "titel": f"ID: {str(pool_person.get('face_id', 'pool')).upper()}",
                "geschlecht": deepface_data.get("Geschlecht", "Unbekannt"),
                "augen": "Braun",
                "stimmung": deepface_data.get("Emotion", "Neutral"),
                "alter": str(deepface_data.get("Alter", "N/A")),
                "gefahr": self._calculate_danger(deepface_data.get("Emotion", "Neutral")),
                "beschreibung": description_data.get("description", "Keine Beschreibung gefunden."),
                "face_image_path": pool_person.get("face_image_path"),
                "source": pool_person.get("source", "pool"),
            })

        for index, person in enumerate(personen_daten, start=1):
            person["titel"] = f"ID: FACE{index}"
        return personen_daten

    def _load_person_data_from_final(self):
        """
        Liest die Anzeige-Daten direkt aus dem final-Ordner.
        Der Pipeline-Thread dient hier nur noch als Synchronisationssignal, dass
        der Batch vollstaendig vorliegt.
        """
        expected_face_count = self.description_repo.read_faces_log_count()
        if expected_face_count and expected_face_count > 0:
            face_indices = list(range(1, expected_face_count + 1))
        else:
            face_indices = self.description_repo.list_face_indices()

        personen_daten = [self._build_person_dict_from_final(face_index) for face_index in face_indices]
        return self._append_pool_people(personen_daten)

    # =========================================================
    # Pipeline-Ergebnis verarbeiten
    # =========================================================

    @pyqtSlot(str, list)
    def handle_pipeline_result(self, status, personen_daten):
        """
        Callback: wird vom PipelineWorker aufgerufen wenn Ergebnisse bereit sind.

        Bei leerem Ergebnis: Akte schließen.
        Bei gültigem Ergebnis: Lade-Anzeige zeigen und Daten aufbereiten.

        :param status: "BATCH" = Ergebnisse vorhanden, "EMPTY" = keine Person erkannt.
        :param personen_daten: Liste mit Personen-Dicts.
        """
        if status == "EMPTY" or not personen_daten:
            # Keine Person erkannt → direkt zurück zum Ausgangszustand
            self._auto_close_monitoring_pending = False
            self._auto_close_monitoring_enabled = False
            self.hide_loading_indicator()
            self.close_folder(reason="empty_result")
            return
        self._set_state(GUIState.RESULTS_READY)
        # Besucherstatistik: nur echte Personen zählen (nicht Pool-Personen).
        # Im Developer-Mode wird nicht gezählt, damit Testläufe die Zahlen nicht verfälschen.
        real_face_count = self.description_repo.read_faces_log_count() or 0
        self._stats_svc.record_session(real_face_count, developer_mode=self.developer_mode)
        self._auto_close_monitoring_pending = True
        self.handle_new_dataset(self._load_person_data_from_final())

    @pyqtSlot(list)
    def handle_new_dataset(self, personen_daten):
        """
        Verarbeitet neue Personendaten und startet die Darstellung.

        Wenn die Mappe schon offen ist: Flip-Animation.
        Wenn die Mappe zu ist: Öffnungs-Animation.
        Wenn gerade animiert wird: kurz warten und wiederholen.

        :param personen_daten: Liste mit Personen-Dicts.
        """
        if not personen_daten:
            return

        self.person_data = self._normalize_person_data(personen_daten)
        self.show_loading_indicator()

        if self.is_animating:
            # Animation läuft → in 100ms erneut versuchen
            QTimer.singleShot(100, lambda: self.handle_new_dataset(personen_daten))
            return

        if self._is_open:
            self.show_flip_video()
        else:
            self.start_animation()

    def update_descriptions_from_files(self):
        """
        Liest fertige Beschreibungen aus den KI-Ausgabe-Dateien und aktualisiert die Container.
        Wird aufgerufen wenn die Mappe geöffnet wird und neue Daten vorliegen.
        Unterstützt sowohl Ollama- als auch Moondream-Beschreibungen.
        """
        if not self.description_repo.exists():
            return
        if not self.active_containers:
            return

        # Entscheiden ob Ollama- oder Moondream-Texte verwendet werden
        from constants import PipelineStage
        ollama_enabled = bool((self._get_pipeline_entry(PipelineStage.OLLAMA) or {}).get("enabled", False))

        for i, container in enumerate(self.active_containers):
            if container._last_description_source:
                continue

            new_text = "Keine Daten gefunden. Akte ausstehend."
            if ollama_enabled:
                description = self.description_repo.read_ollama_description(i)
            else:
                description = self.description_repo.read_moondream_description(i)
            if description:
                new_text = description

            if new_text != container._last_description_source:
                container._last_description_source = new_text
                translated_text = self.translator.translate_text(new_text) if self.translator else new_text
                if container.beschreibung.full_text != translated_text:
                    container.beschreibung.full_text = translated_text
                    container.beschreibung.start_typing()

            # DeepFace-Daten (Emotion, Alter, Geschlecht) aktualisieren
            deepface_data = self.description_repo.read_deepface_data(i) or {}
            emotion = deepface_data.get("Emotion") or deepface_data.get("emotion")
            age = deepface_data.get("Alter") or deepface_data.get("alter")
            gender = deepface_data.get("Geschlecht") or deepface_data.get("geschlecht")
            deepface_signature = (emotion, age, gender)

            if deepface_signature != container._last_deepface_source and any(deepface_signature):
                container._last_deepface_source = deepface_signature
                container.update_stats_from_deepface(emotion=emotion, age=age, gender=gender)

    # =========================================================
    # UI-Elemente für offene Mappe
    # =========================================================

    def setup_ui_elements(self):
        """
        Erstellt PersonContainer für alle Personen in person_data.
        Positioniert Container auf der offenen Mappe (2x2 Raster).
        Lädt Sketch-Bilder aus dem Sketch-Verzeichnis.
        """
        self.active_containers = []
        # Feste Positionen der vier Personen-Karten auf der offenen Mappe
        pos_list = [(230, 50), (1000, 50), (230, 540), (1000, 540)]
        for i, pos in enumerate(pos_list):
            if i < len(self.person_data):
                container = PersonContainer(
                    self.person_data[i],
                    i,
                    self.current_language,
                    sound_service=self._sound_svc,
                    developer_mode=self.developer_mode,
                )
                description = self.person_data[i].get("beschreibung")
                if description:
                    container._last_description_source = description
                    translated = self.translator.translate_text(description) if self.translator else description
                    container.beschreibung.full_text = translated
                    container.beschreibung.start_typing()

                # Sketch-Bild laden (aus Sketch-Ordner oder direkt vom face_image_path)
                image_path = self.person_data[i].get("face_image_path")
                if not image_path or not os.path.exists(image_path):
                    image_path = os.path.join(PATHS["sketch_dir"], f"face{i + 1}.png")
                if os.path.exists(image_path):
                    sketch_img = create_advanced_sketch(image_path)
                    container.set_sketch_image(sketch_img)

                # Container für Maus-Events transparent (nur Anzeige, keine Interaktion)
                container.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
                proxy = self.scene.addWidget(container)
                proxy.setPos(pos[0], pos[1])
                proxy.setZValue(1)
                self.active_containers.append(container)

    def setup_buttons(self):
        """
        Erstellt Sprach- und Reset-Button für die offene Mappe.
        Positionen: rechts oben (Sprache) und darunter (Reset).
        """
        button_scale = 0.1
        right_margin = 40
        top_margin = 80
        vertical_gap = 150

        # Sprach-Button (DE/EN Umschalter)
        self.btn_language = AnimatedGraphicsButton(
            PATHS["language_de"], PATHS["language_en"], scale=button_scale
        )
        w1 = self.btn_language.pixmap1.width() * button_scale
        self.btn_language.setPos(SCENE_WIDTH - right_margin - w1, top_margin)
        self.btn_language.setZValue(100)
        self.scene.addItem(self.btn_language)
        # Button-Zustand auf aktuelle Sprache setzen
        if self.current_language == "en":
            self.btn_language.is_toggled = True
            self.btn_language.current_pixmap = self.btn_language.pixmap2
            self.btn_language.update()
        self._set_language_button_enabled_state(not self._language_button_cooldown_timer.isActive())
        self.btn_language.clicked.connect(self._on_language_button_clicked)

        # Reset-Button (Akte schließen)
        self.btn_reset = AnimatedGraphicsButton(self._reset_button_pixmap_path, scale=button_scale)
        w2 = self.btn_reset.pixmap1.width() * button_scale
        self.btn_reset.setPos(SCENE_WIDTH - right_margin - w2, top_margin + vertical_gap)
        self.btn_reset.setZValue(100)
        self.scene.addItem(self.btn_reset)
        self.btn_reset.clicked.connect(self.reset_logic)

    # =========================================================
    # Lade-Anzeige
    # =========================================================

    def show_loading_indicator(self):
        """
        Zeigt einen Lade-Spinner an während die KIs arbeiten.

        Wenn die Mappe offen ist: Spinner über dem Reset-Button.
        Wenn die Mappe zu ist: großer Spinner unten rechts.
        Startet gleichzeitig den Pipeline-Timeout-Timer.
        """
        if self.state != GUIState.RESULTS_READY:
            self._set_state(GUIState.ANALYZING)
        self.loading_active = True
        self._auto_close_monitoring_enabled = False
        if self.state != GUIState.RESULTS_READY:
            self._auto_close_monitoring_pending = False
        self._start_pipeline_timeout()

        # Kamerabild einfrieren damit der Nutzer das aufgenommene Bild sieht
        if not self._is_open:
            self._freeze_camera_preview()

        # Vorhandenen Spinner entfernen
        if self.loading_item is not None:
            if hasattr(self.loading_item, "stop"):
                self.loading_item.stop()
            self.loading_item.hide()
            self.scene.removeItem(self.loading_item)
            self.loading_item = None

        if self._is_open and hasattr(self, "btn_reset"):
            # Kleiner Spinner über dem Reset-Button (Mappe offen)
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
            # Großer Spinner (Mappe geschlossen)
            diameter = 240
            self.loading_item = LoadingSpinnerItem(diameter=diameter)
            self.scene.addItem(self.loading_item)
            self.loading_item.setPos(SCENE_WIDTH - diameter - 450, SCENE_HEIGHT - diameter - 120)

    def hide_loading_indicator(self):
        """
        Entfernt den Lade-Spinner und stoppt den Timeout-Timer.
        Wird aufgerufen wenn Ergebnisse vorliegen oder die Akte geschlossen wird.
        """
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
        """
        Startet den Timeout-Wächter für die Pipeline.
        Wenn die KIs nach pipeline_timeout_seconds Sekunden keine Ergebnisse liefern,
        wird die Akte automatisch geschlossen.
        """
        timeout_ms = max(1, int(self.pipeline_timeout_seconds)) * 1000
        self._pipeline_timeout_timer.start(timeout_ms)

    def _on_pipeline_timeout(self):
        """
        Callback: Pipeline-Timeout abgelaufen.
        Schließt die Akte und gibt eine Debug-Meldung aus.
        """
        if not self.loading_active:
            return
        if self.developer_mode:
            print(f"Pipeline timeout after {self.pipeline_timeout_seconds} seconds. Returning to closed folder.")
        self.hide_loading_indicator()
        self.close_folder(reason="pipeline_timeout")

    # =========================================================
    # Reset-Button visueller Zustand
    # =========================================================

    def _set_reset_button_loading(self, is_loading):
        """
        Tauscht Reset-Button-Bild gegen leere Variante während KIs arbeiten.
        :param is_loading: True = leere Variante, False = Original wiederherstellen.
        """
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
        """
        Lässt den Reset-Button pulsieren wenn Auto-Close bevorsteht.
        :param is_warning: True = pulsieren, False = Normal-Zustand.
        """
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

    # =========================================================
    # Vollbild
    # =========================================================

    def _set_fullscreen(self, enabled):
        """
        Schaltet Vollbild ein/aus.
        Im Vollbild wird der Fensterrahmen entfernt (FramelessWindowHint).
        :param enabled: True = Vollbild, False = normales Fenster.
        """
        self.is_fullscreen = bool(enabled)
        # FramelessWindowHint dynamisch setzen, damit kein Rahmen im Vollbild erscheint
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, self.is_fullscreen)
        if self.is_fullscreen:
            self.showFullScreen()
        else:
            self.showNormal()

    def apply_window_state(self):
        """Wendet den gespeicherten Fensterzustand (Vollbild ja/nein) an."""
        self._set_fullscreen(self.is_fullscreen)

    # =========================================================
    # Fenster-Events
    # =========================================================

    def keyPressEvent(self, event):
        """
        Tastatur-Handler: E öffnet/schließt das Admin-Menü.
        Im Museumsbetrieb ohne Tastatur nicht erreichbar.
        """
        if event.key() == Qt.Key.Key_E:
            if self.admin_menu.isVisible():
                self.admin_menu.hide()
            else:
                self._sync_admin_menu_with_config()
                self.admin_menu.update_geometry(self.size())
                self.admin_menu.show()
                self.admin_menu.raise_()
        super().keyPressEvent(event)

    def resizeEvent(self, event):
        """
        Skaliert die Szene beim Fenster-Resize und passt Admin-Menü-Geometrie an.
        Wichtig für den Vollbild-Wechsel.
        """
        super().resizeEvent(event)
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        if self.admin_menu.isVisible():
            self.admin_menu.update_geometry(self.size())
