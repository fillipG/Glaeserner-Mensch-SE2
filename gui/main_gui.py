"""
DATEI: main_gui.py
BESCHREIBUNG: Haupt-GUI der "Akte" mit Kamera-Integration, Animationen, Ladeanzeige und automatischem Schließen bei Abwesenheit.
- Verarbeitet neue Datensätze und Pipeline-Ergebnisse.
- Zeigt Kamera-Feed mit Gesichtserkennung und Bounding-Boxes.
- Verwendet Timer für Fotoverzögerung, Anwesenheitsüberwachung und Pipeline-Timeouts.
- Integriert Admin-Menü für LLM-Auswahl und andere Einstellungen.
- Nutzt Übersetzung für mehrsprachige KI-Beschreibungen.
- Verwaltet GUI-Zustände (geschlossen, offen, analysierend, Ergebnisse bereit).
- Bereinigt Pipeline-Ausgabeverzeichnisse bei Bedarf.
AUTOR: Fillip Giffhorn
"""

import os
import sys
import cv2
import time
import shutil

from PyQt6.QtWidgets import (QApplication, QGraphicsView, QGraphicsScene,
                             QLabel, QFrame, QPushButton, QMessageBox)
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

# Definition von auswählbaren Ollama modellen
LLM_OPTIONS = [
    {"label": "Ollama - qwen3 4b (aktuell)", "value": "qwen3:4b"},
    {"label": "Ollama - gemma3 1b (aktuell, leicht)", "value": "gemma3:1b"},
    {"label": "Ollama - gemma3 4b (aktuell, stark)", "value": "gemma3:4b"},
    {"label": "Ollama - qwen2.5 3b (empfohlen)", "value": "qwen2.5:3b"},
    {"label": "Ollama - llama3.2 3b", "value": "llama3.2:3b"},
    {"label": "Ollama - llama3.2 1b (schnell)", "value": "llama3.2:1b"},
    {"label": "Ollama - gemma3", "value": "gemma3"},
    {"label": "Ollama - phi3 3.8b", "value": "phi3:3.8b"},
]

# --- HAUPT GUI ---
class ScalingAkteGUI(QGraphicsView):
    """
    Haupt-GUI der "Akte" mit Kamera-Integration, Animationen, Ladeanzeige und automatischem Schließen bei Abwesenheit.
    """
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
        self.person_data = []
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
        self._clear_pipeline_outputs_on_close = False
        self.btn_open = None

        # Cooldown fuer den Sprachwechsel-Button
        self._language_button_cooldown_ms = 5_000
        self._language_button_cooldown_timer = QTimer(self)
        self._language_button_cooldown_timer.setSingleShot(True)
        self._language_button_cooldown_timer.timeout.connect(self._on_language_button_cooldown_timeout)

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

        # WARNUNGEN: "Keine Person" Timer
        self._no_person_warning_timer = QTimer(self)
        self._no_person_warning_timer.timeout.connect(self._blink_no_person_warning)

        # FACE-OVERLAY: Bounding-Boxes im Live-Preview
        self._face_cascade = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )
        self._face_detect_interval_ms = 100 #Update der Bounding-Boxes in Live-Kamera
        self._last_face_detect_ms = 0
        self._last_faces = []
        self._face_detection_scale = 0.5
        self.camera_pixmap_item = None
        self._last_camera_preview_pixmap = None
        self._pipeline_timeout_timer = QTimer(self)
        self._pipeline_timeout_timer.setSingleShot(True)
        self._pipeline_timeout_timer.timeout.connect(self._on_pipeline_timeout)

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
        """
        Normalisiert die Personendaten und begrenzt sie auf maximal 4 Personen
        :param personen_daten: Die rohen Personendaten, die von der Pipeline oder dem Pool kommen können. Erlaubt sind Listen oder andere iterierbare Strukturen.
        :return: Eine Liste von maximal 4 normalisierten Personendatensätzen.
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

    def _set_state(self, state):
        """
        Setzt den internen GUI-Zustand.
        :param state: Neuer GUIState.
        :return: None
        """
        self.state = state

    def _freeze_camera_preview(self):
        """
        Setzt die Kamera-Vorschau auf ein statisches Bild, um als Anwender zu sehen, welches Bild gerade analysiert wird
        """
        if self.camera_pixmap_item is None:
            return
        if self._last_camera_preview_pixmap is not None:
            self.camera_pixmap_item.setPixmap(self._last_camera_preview_pixmap)
            return
        placeholder = QPixmap(self._cam_display_w, self._cam_display_h)
        placeholder.fill(QColor("black"))
        self.camera_pixmap_item.setPixmap(placeholder)

    def show_loading_indicator(self):
        """
        Zeigt ein Lade-Symbol je nach GUI-Zustand an.
        """
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
        """
        Beendet das Ladesymbol.
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
        Startet den Timer, der die Pipeline-Timeout-Überwachung aktiviert. Wenn der Timer abläuft, wird die Pipeline als "hängen geblieben" betrachtet und die Akte wird geschlossen.
        :return:
        """
        timeout_ms = max(1, int(self.pipeline_timeout_seconds)) * 1000
        self._pipeline_timeout_timer.start(timeout_ms)

    def _on_pipeline_timeout(self):
        """
        Callback, wenn die Pipeline-Timeout-Überwachung auslöst. Schließt die Akte mit einem entsprechenden Grund und versteckt die Ladeanzeige.
        """
        if not self.loading_active:
            return
        if self.developer_mode:
            print(f"Pipeline timeout after {self.pipeline_timeout_seconds} seconds. Returning to closed folder.")
        self.hide_loading_indicator()
        self.close_folder(reason="pipeline_timeout")

    def _set_reset_button_loading(self, is_loading):
        """
        Zeigt auf dem Reset-Button ein ladesymbol an, indem das Originalbild durch ein leeres Bild ersetzt wird. Beim Beenden des Ladevorgangs wird das Originalbild wiederhergestellt.
        :param is_loading: True, um den Ladezustand anzuzeigen, False, um zum Normalzustand zurückzukehren.
        :return:
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
        Lässt reset Button pulsieren, wenn keine Person erkannt wird und die Akte kurz davor ist, sich automatisch zu schließen. Beim Beenden der Warnung wird der Button-Zustand wiederhergestellt.
        :param is_warning: True, um den Warnzustand anzuzeigen, False, um zum Normalzustand zurückzukehren.
        :return:
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

    def _stop_no_person_timer(self):
        """
        Stoppt den Timer für die "Keine Person erkannt" Warnung und setzt den Zähler der verpassten Anwesenheitsprüfungen zurück.
        Außerdem wird die Reset-Button-Warnung deaktiviert, falls sie aktiv ist.
        """
        if self._no_person_warning_timer.isActive():
            self._no_person_warning_timer.stop()
        self._missed_presence_checks = 0
        self._set_reset_button_warning(False)

    def _get_auto_close_missed_check_limit(self):
        """
        Berechnet die Anzahl der verpassten Anwesenheitsprüfungen, die erforderlich sind, bevor die Akte automatisch geschlossen wird. Diese Berechnung basiert auf den Einstellungen für das Intervall der Anwesenheitsprüfungen und die Zeit bis zum automatischen Schließen.
        :return: Die Anzahl der verpassten Anwesenheitsprüfungen, die zum automatischen Schließen führen.
        """
        interval_ms = max(100, int(self.no_person_check_interval_ms))
        timeout_ms = max(5000, int(self.close_on_no_person_seconds) * 1000)
        return max(1, int(round(timeout_ms / float(interval_ms))))

    def _get_warning_start_missed_checks(self):
        """
        Berechnet die Anzahl der verpassten Anwesenheitsprüfungen, bei der die Warnung "Keine Person erkannt" aktiviert wird. Diese Berechnung basiert auf der Anzahl der verpassten Prüfungen, die zum automatischen Schließen führen, minus einer Sicherheitsmarge von 5 Sekunden.
        :return: Die Anzahl der verpassten Anwesenheitsprüfungen, bei der die Warnung aktiviert wird.
        """
        interval_ms = max(100, int(self.no_person_check_interval_ms))
        warning_checks = max(1, int(round(5000 / float(interval_ms))))
        return max(0, self._get_auto_close_missed_check_limit() - warning_checks)

    def _update_no_person_warning_state(self):
        """
        Aktualisiert den Zustand der "Keine Person erkannt" Warnung basierend auf der Anzahl der verpassten Anwesenheitsprüfungen. Wenn die Anzahl der verpassten Prüfungen den Schwellenwert für die Warnung erreicht oder überschreitet, wird die Warnung aktiviert und der Reset-Button beginnt zu pulsieren. Wenn die Anzahl der verpassten Prüfungen unter den Schwellenwert fällt, wird die Warnung deaktiviert und der Reset-Button kehrt zum Normalzustand zurück.
        :return:
        """
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
        """
        Lässt den Reset-Button pulsieren, um anzuzeigen, dass keine Person erkannt wurde und die Akte kurz davor ist, sich automatisch zu schließen. Diese Funktion wird von einem Timer aufgerufen, der alle 400 ms ausgelöst wird, wenn die Warnung aktiv ist.
        :return:
        """
        if not self._is_open or self.loading_active or self.is_animating:
            self._stop_no_person_timer()
            return
        self._set_reset_button_warning(True)

    def close_folder(self, reason: str = "", animated: bool = True):
        """
        Schließt die Akte mit optionaler Animation. Je nach Grund des Schließens können zusätzliche Aktionen wie das Bereinigen von Pipeline-Ausgabeverzeichnissen ausgelöst werden.
        :param reason: Der Grund für das Schließen der Akte. Mögliche Werte: "manual", "manual_countdown", "auto_close", "pipeline_timeout", "empty_result". Je nach Grund können bestimmte Aktionen ausgelöst werden, z.B. das Bereinigen von Pipeline-Ausgabeverzeichnissen.
        :param animated: True, um eine Schließ-Animation abzuspielen, bevor die Akte geschlossen wird. False, um die Akte sofort zu schließen, ohne Animation.
        :return:
        """
        self._clear_pipeline_outputs_on_close = reason in {
            "manual",
            "manual_countdown",
            "auto_close",
            "pipeline_timeout",
            "empty_result",
        }
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

    def _clear_pipeline_output_dirs(self):
        """
        Entfernt alle Dateien und Unterordner in den Pipeline-Ausgabeverzeichnissen, um sicherzustellen, dass der nächste Durchlauf mit einem sauberen Zustand beginnt. Diese Funktion wird bewusst beim Schließen der Akte aufgerufen, um zu verhindern, dass veraltete Dateien aus vorherigen Durchläufen die aktuelle Verarbeitung stören.
        :return:
        """
        # Beim bewussten Ruecksprung werden alte Pipeline-Ergebnisse entfernt,
        # damit der naechste Durchlauf nicht auf Restdateien aus dem vorherigen Batch trifft.
        cleanup_dirs = [
            PATHS["final_dir"],
            os.path.join("General ordner", "ollama_ai", "ollama_inbox"),
        ]
        for folder in cleanup_dirs:
            if not os.path.exists(folder):
                continue
            for entry in os.listdir(folder):
                entry_path = os.path.join(folder, entry)
                try:
                    if os.path.isfile(entry_path) or os.path.islink(entry_path):
                        os.unlink(entry_path)
                    elif os.path.isdir(entry_path):
                        shutil.rmtree(entry_path)
                except Exception as exc:
                    if self.developer_mode:
                        print(f"Close-Cleanup konnte {entry_path} nicht loeschen: {exc}")

    @pyqtSlot(list)
    def handle_new_dataset(self, personen_daten):
        """
        Verarbeitet einen neuen Satz von Personendaten, der von der Pipeline oder dem Pool bereitgestellt wird. Je nach aktuellem Zustand der GUI und ob bereits eine Akte geöffnet ist, wird entweder die Flip-Video-Animation abgespielt oder direkt die Analyse gestartet. Wenn bereits eine Animation läuft, wird die Verarbeitung des neuen Datensatzes um 100 ms verzögert, um Konflikte zu vermeiden.
        :param personen_daten: Eine Liste von Personendatensätzen, die die Informationen über die erkannten Personen enthalten.
        """
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
        """
        Callback-Funktion, die aufgerufen wird, wenn die Pipeline ein Ergebnis zurückgibt. Je nach Status und Inhalt der Personendaten wird entweder die Akte geschlossen oder die Ergebnisse verarbeitet und angezeigt.
        :param status:
        :param personen_daten:
        :return:
        """
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
        """
        Scannt den 'final' Ordner und extrahiert die (ggf. mehrzeilige) 'description'.
        """
        if not self.description_repo.exists():
            return

        if not self.active_containers:
            return

        ollama_enabled = bool((self._get_pipeline_entry("ollama") or {}).get("enabled", False))

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

            deepface_data = self.description_repo.read_deepface_data(i) or {}
            emotion = deepface_data.get("Emotion") or deepface_data.get("emotion")
            age = deepface_data.get("Alter") or deepface_data.get("alter")
            gender = deepface_data.get("Geschlecht") or deepface_data.get("geschlecht")
            deepface_signature = (emotion, age, gender)

            if deepface_signature != container._last_deepface_source and any(deepface_signature):
                container._last_deepface_source = deepface_signature
                container.update_stats_from_deepface(emotion=emotion, age=age, gender=gender)

    def _refresh_descriptions_for_language(self):
        """
        Aktualisiert die angezeigten Beschreibungen in der GUI, wenn die Sprache geändert wird. Diese Funktion wird aufgerufen, nachdem die Übersetzungsfunktion die neuen Texte generiert hat, um sicherzustellen, dass die angezeigten Beschreibungen mit der aktuellen Sprache übereinstimmen.
        :return:
        """
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
        """
        Callback-Funktion, die aufgerufen wird, wenn ein neues Kamera-Frame verfügbar ist. Verarbeitet das Frame, führt Gesichtserkennung durch und aktualisiert die Kamera-Vorschau in der GUI mit den erkannten Gesichtern als Bounding-Boxes.
        :param frame: Das aktuelle Kamera-Frame, das verarbeitet und in der GUI angezeigt werden soll.
        :return:
        """
        if self.camera_pixmap_item is None:
            return
        try:
            # KAMERA-FRAME: Face-Boxes + Rendering
            display_frame = frame
            if self._face_cascade is not None and not self._face_cascade.empty():
                now_ms = int(time.perf_counter() * 1000)
                if (now_ms - self._last_face_detect_ms) >= self._face_detect_interval_ms:
                    self._last_face_detect_ms = now_ms
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    scale = max(0.25, min(1.0, float(self._face_detection_scale)))
                    if scale < 1.0:
                        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    else:
                        small = gray
                    faces = self._face_cascade.detectMultiScale(
                        small,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30),
                    )
                    if scale < 1.0 and len(faces) > 0:
                        faces = [
                            (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
                            for (x, y, w, h) in faces
                        ]
                    self._last_faces = faces

                faces = self._last_faces or []
                if len(faces) > 0:
                    display_frame = frame.copy()
                    box_color = (188, 228, 244)  # BGR for #f4e4bc
                    for (x, y, w, h) in faces:
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_color, 2)

            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
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
        """
        Callback-Funktion, die aufgerufen wird, wenn sich der Anwesenheitsstatus einer Person ändert. Wenn eine Person erkannt wird, wird der Timer für die "Keine Person erkannt" Warnung gestoppt. Wenn keine Person erkannt wird und die Akte geöffnet ist, wird die Anzahl der verpassten Anwesenheitsprüfungen erhöht und je nach Anzahl der verpassten Prüfungen entweder eine Warnung aktiviert oder die Akte automatisch geschlossen.
        :param is_present: True, wenn eine Person erkannt wird, False wenn keine Person erkannt wird.
        :return:
        """
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
        """
        Zeigt die geschlossene Mappe an, indem die Szene bereinigt und das geschlossene Mappe-Bild geladen wird. Es werden auch die Logos und die Kamera-Vorschau mit Rahmen hinzugefügt. Alle relevanten Timer und Zustände werden zurückgesetzt, um sicherzustellen, dass die GUI bereit ist für den nächsten Durchlauf.
        :return:
        """
        if self._clear_pipeline_outputs_on_close:
            self._clear_pipeline_output_dirs()
            self._clear_pipeline_outputs_on_close = False
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

        # Statische Logos hinzufügen
        logo_configs = [
            {
                "path": PATHS.get("logo_bmftr"),
                "scale": 0.4,  
                "pos": (30, SCENE_HEIGHT - 30),
            },
            {
                "path": PATHS.get("logo_ki_owl"),
                "scale": 0.11,
                "pos": (400, SCENE_HEIGHT - 30),
            },
            {
                "path": PATHS.get("logo_th_owl"),
                "scale": 0.5,
                "pos": (30, 200),
            }
        ]
        for cfg in logo_configs:
            logo_path = cfg.get("path")
            if not logo_path or not os.path.exists(logo_path):
                continue
            pixmap = QPixmap(logo_path)
            if pixmap.isNull():
                continue
            logo_item = self.scene.addPixmap(pixmap)
            scale = float(cfg.get("scale") or 1.0)
            if scale <= 0:
                scale = 1.0
            logo_item.setScale(scale)
            pos_x, pos_bottom_y = cfg.get("pos", (30, SCENE_HEIGHT - 30))
            scaled_height = pixmap.height() * scale
            logo_item.setPos(pos_x, pos_bottom_y - scaled_height)
            logo_item.setZValue(8)

        # KAMERA: Vorschau in der GUI 
        self._cam_display_w = 896
        self._cam_display_h = 504
        cam_x = 20
        cam_y = ((SCENE_HEIGHT - self._cam_display_h) // 2)-30

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
        #Hier endet erstmal die Kamera- und Timer-Setup-Phase

        self.btn_open = None
        if self.developer_mode:
            self.btn_open = QPushButton("Mappe öffnen")
            self.btn_open.setFixedSize(300, 80)
            self.btn_open.setStyleSheet(
                "QPushButton { background-color: #3d2b1f; color: #f4e4bc; border: 3px solid #f4e4bc; border-radius: 15px; font-family: 'Graduate'; font-size: 24px; font-weight: bold; } QPushButton:hover { background-color: #5a4030; }")
            self.btn_open.clicked.connect(self.show_animation_with_timer)
            proxy = self.scene.addWidget(self.btn_open)
            proxy.setPos(550, 100)

        self.folder_closed.emit()

    def show_animation_with_timer(self):
        """
        Entscheidet, ob das Öffnen der Mappe sofort mit einer Animation erfolgen soll oder ob zuerst ein Timer angezeigt wird, basierend auf dem aktuellen Zustand der GUI und der Sichtbarkeit des Timer-Items. Wenn bereits eine Animation läuft oder aktive Container vorhanden sind, wird die Funktion ohne Aktion verlassen. Wenn kein Timer-Item vorhanden ist, wird die Animation sofort gestartet. Wenn das Timer-Item existiert, aber nicht sichtbar ist (z.B. weil es versehentlich geschlossen wurde), wird es entfernt und die Animation wird gestartet. Andernfalls wird der Timer gestartet, um die verbleibende Zeit bis zum automatischen Öffnen anzuzeigen.
        :return:
        """
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
        """
        Startet den Timer, der die verbleibende Zeit bis zum automatischen Öffnen der Mappe anzeigt. Wenn die Entwickleroption aktiviert ist, wird die Animation sofort gestartet, ohne den Timer anzuzeigen.
        :return:
        """
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
        btn_open = getattr(self, "btn_open", None)
        if btn_open is not None:
            btn_open.setEnabled(False)
        self.wait_timer.start(33)

    def _update_wait_timer(self):
        """
        Aktualisiert den Fortschritt des Timers, der die verbleibende Zeit bis zum automatischen Öffnen der Mappe anzeigt. Berechnet die verstrichene Zeit seit dem Start des Timers und aktualisiert das Timer-Item entsprechend. Wenn die verbleibende Zeit abgelaufen ist, wird der Timer gestoppt, das Timer-Item ausgeblendet und die Animation zum Öffnen der Mappe gestartet.
        :return:
        """
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
            btn_open = getattr(self, "btn_open", None)
            if btn_open is not None:
                btn_open.setEnabled(True)
            self.start_animation()

    def start_animation(self, checked=False, video_path=PATHS["open_animation"], end_callback=None):
        """
    Startet die Animation zum Öffnen oder Schließen der Mappe, abhängig von den übergebenen Parametern. Wenn bereits eine Animation läuft, wird die Funktion ohne Aktion verlassen. Wenn der Parameter 'checked' ein String oder Pfad ist, wird dieser als 'video_path' interpretiert und 'checked' wird auf False gesetzt. Je nach End-Callback und aktuellem Zustand der Mappe wird entweder die Flip-Animation oder die Öffnungsanimation abgespielt. Wenn die angegebene Videodatei nicht existiert, wird stattdessen die Funktion zum Anzeigen der offenen Mappe aufgerufen.
        :param checked: Ein boolescher Wert oder ein String/Pfad. Wenn es ein String oder Pfad ist, wird er als 'video_path' interpretiert und 'checked' wird auf False gesetzt. Wenn es ein boolescher Wert ist, steuert er die Auswahl der Animation basierend auf dem End-Callback und dem aktuellen Zustand der Mappe.
        :param video_path:
        :param end_callback:
        :return:
        """
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
        """
        Aktualisiert das aktuelle Frame der laufenden Animation, indem es das nächste Frame aus der Videodatei liest, es in ein QPixmap umwandelt und in der Szene anzeigt. Wenn das Ende des Videos erreicht ist, wird die Animation gestoppt, die Videodatei freigegeben und der End-Callback aufgerufen. Wenn während der Animation ein Schließvorgang angefordert wird, wird dieser nach Abschluss der Animation ausgeführt.
        :return:
        """
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
        """
        Zeigt die offene Mappe an, indem die Szene bereinigt und das offene Mappe-Bild geladen wird. Es werden auch die UI-Elemente für die angezeigten Personen eingerichtet und die Beschreibungen aus den Dateien aktualisiert. Alle relevanten Timer und Zustände werden zurückgesetzt, um sicherzustellen, dass die GUI bereit ist, die Ergebnisse der Pipeline anzuzeigen und auf Anwesenheitsänderungen zu reagieren.
        :return:
        """
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
        """
        Spielt das Umblättern-Video ab und kehrt danach zur offenen Mappe zurück.
        """
        self._set_state(GUIState.FLIPPING)
        self.start_animation(PATHS["flip_animation"], end_callback=self.show_open_folder)

    def setup_ui_elements(self):
        """
        Richtet die UI-Elemente für die angezeigten Personen ein, basierend auf den bereitgestellten Personendaten. Es werden Container für jede erkannte Person erstellt, die Beschreibungen aktualisiert und die Gesichtsbilder als Skizzen dargestellt. Die Container werden in der Szene positioniert und für Mausinteraktionen deaktiviert, um sicherzustellen, dass sie nur zur Anzeige von Informationen dienen.
        :return:
        """
        self.active_containers = []
        pos_list = [(230, 50), (1000, 50), (230, 540), (1000, 540)]
        for i, pos in enumerate(pos_list):
            if i < len(self.person_data):
                container = PersonContainer(
                    self.person_data[i],
                    i,
                    self.current_language,
                    developer_mode=self.developer_mode,
                )
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
        """
        Richtet die Schaltflächen für die Sprachumschaltung und das Zurücksetzen der Akte ein. Die Schaltflächen werden in der Szene positioniert, mit den entsprechenden Grafiken versehen und mit den entsprechenden Callback-Funktionen verbunden, um die gewünschten Aktionen auszuführen, wenn sie angeklickt werden.
        :return:
        """
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
        self._set_language_button_enabled_state(not self._language_button_cooldown_timer.isActive())
        self.btn_language.clicked.connect(self._on_language_button_clicked)

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
        """Speichert die aktuelle Konfiguration zurück in die config.yaml."""
        self.config_service.save(self.config)

    def _ensure_config_defaults(self, config):
        """Stellt sicher, dass alle erforderlichen Standardwerte in der Konfiguration vorhanden sind, um eine konsistente und vollständige Konfiguration zu gewährleisten. Diese Funktion wird aufgerufen, nachdem die Konfiguration geladen wurde, um sicherzustellen, dass alle fehlenden Werte mit den Standardwerten aus der ConfigService ergänzt werden."""
        self.config_service.ensure_defaults(config)

    def _get_pipeline_entry(self, model_id):
        """Sucht in der Pipeline-Konfiguration nach einem Eintrag mit der angegebenen ID und gibt diesen zurück. Wenn kein Eintrag mit der angegebenen ID gefunden wird, wird None zurückgegeben. Diese Funktion wird verwendet, um die spezifischen Einstellungen für verschiedene Modelle oder Komponenten in der Pipeline zu verwalten und zu aktualisieren."""
        pipeline = self.config.setdefault("pipeline", [])
        return next((p for p in pipeline if p.get("id") == model_id), None)

    def _update_config_value(self, key, value):
        """Updatet Werte aus config.yaml"""
        self.config[key] = value
        self._save_config()

    def _get_face_yolo_confidence(self):
        """Liest den Konfidenzwert für die Gesichterkennung aus"""
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
        """Updatet Werte in der Pipeline-Konfiguration für ein bestimmtes Modell oder eine bestimmte Komponente, identifiziert durch die model_id. Wenn der Eintrag für die angegebene model_id nicht existiert, wird zuerst sichergestellt, dass die Standardwerte in der Konfiguration vorhanden sind, und dann wird der Eintrag erneut gesucht. Wenn der Eintrag gefunden wird, wird der angegebene Schlüssel mit dem neuen Wert aktualisiert und die Konfiguration wird gespeichert. Diese Funktion ermöglicht es, spezifische Einstellungen für verschiedene Modelle oder Komponenten in der Pipeline dynamisch zu aktualisieren."""
        entry = self._get_pipeline_entry(model_id)
        if entry is None:
            self.config_service.ensure_defaults(self.config)
            entry = self._get_pipeline_entry(model_id)
        if entry is None:
            return
        entry[key] = value
        self._save_config()

    def _update_pool_value(self, key, value):
        """Updatet Werte in der Pool-Konfiguration, die für die Verwaltung von zusätzlichen Personen in der Pipeline verwendet wird. Es wird sichergestellt, dass der "pool" Abschnitt in der Konfiguration als Dictionary existiert, bevor der angegebene Schlüssel mit dem neuen Wert aktualisiert wird. Nach dem Update wird die Konfiguration gespeichert und die Pool-Einstellungen werden neu geladen, um sicherzustellen, dass die Änderungen sofort wirksam werden. Diese Funktion ermöglicht es, die Einstellungen für den Pool dynamisch zu aktualisieren, ohne dass die gesamte Pipeline-Konfiguration neu geladen werden muss."""
        pool = self.config.setdefault("pool", {})
        if not isinstance(pool, dict):
            pool = {}
            self.config["pool"] = pool
        pool[key] = value
        self._save_config()
        self._reload_pool_settings()

    def _reload_pool_settings(self):
        """Lädt die Pool-Einstellungen neu, indem es die Pipeline auffordert, die Pool-Konfiguration erneut zu laden. Diese Funktion wird aufgerufen, nachdem die Pool-Einstellungen in der Konfiguration aktualisiert wurden, um sicherzustellen, dass die Änderungen sofort wirksam werden. Es wird überprüft, ob die Pipeline existiert und über eine Methode zum Neuladen der Pool-Einstellungen verfügt, bevor der Reload angefordert wird."""
        pipeline = getattr(self, "_pipeline", None)
        if pipeline is not None and hasattr(pipeline, "request_pool_reload"):
            pipeline.request_pool_reload()

    def _reload_pipeline_settings(self):
        """Lädt die Pipeline-Einstellungen neu, indem es die Pipeline auffordert, die gesamte Pipeline-Konfiguration erneut zu laden. Diese Funktion wird aufgerufen, nachdem die Pipeline-bezogenen Einstellungen in der Konfiguration aktualisiert wurden, um sicherzustellen, dass die Änderungen sofort wirksam werden. Es wird überprüft, ob die Pipeline existiert und über eine Methode zum Neuladen der Pipeline-Konfiguration verfügt, bevor der Reload angefordert wird."""
        pipeline = getattr(self, "_pipeline", None)
        if pipeline is not None and hasattr(pipeline, "request_pipeline_reload"):
            pipeline.request_pipeline_reload()

    def _sync_local_ollama_worker_state(self):
        """Synchronisiert den Zustand des lokalen Ollama-Workers, indem es die Pipeline auffordert, den Ollama-Worker-Status zu aktualisieren. Diese Funktion wird aufgerufen, nachdem die Ollama-bezogenen Einstellungen in der Konfiguration aktualisiert wurden, um sicherzustellen, dass der lokale Ollama-Worker mit den neuen Einstellungen übereinstimmt. Es wird überprüft, ob die Pipeline existiert und über eine Methode zum Synchronisieren des Ollama-Worker-Zustands verfügt, bevor die Synchronisierung angefordert wird. Wenn ein Fehler auftritt, wird eine Fehlermeldung angezeigt, um den Benutzer über das Problem zu informieren."""
        worker_manager = getattr(self, "_local_worker_manager", None)
        if worker_manager is None:
            return
        try:
            worker_manager.sync_ollama_worker_state()
        except RuntimeError as exc:
            error_message = (
                "Der lokale Ollama-Worker konnte nicht aktualisiert werden.\n\n"
                f"{exc}"
            )
            print(error_message)
            QMessageBox.critical(self, "Ollama-Start fehlgeschlagen", error_message)

    def _connect_admin_menu(self):
        """Verbindet die Signale des Admin-Menüs mit den entsprechenden Callback-Funktionen, um die Änderungen in den Einstellungen zu verarbeiten. Diese Funktion wird aufgerufen"""
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
        self.admin_menu.ollama_enabled_changed.connect(self._on_ollama_enabled)
        self.admin_menu.ollama_prompt_changed.connect(self._on_ollama_prompt)
        self.admin_menu.deepface_enabled_changed.connect(self._on_deepface_enabled)
        self.admin_menu.deepface_retinaface_changed.connect(self._on_deepface_retinaface_changed)
        self.admin_menu.fer_enabled_changed.connect(self._on_fer_enabled)
        self.admin_menu.llm_model_changed.connect(self._on_llm_model_changed)
        self.admin_menu.reset_defaults_requested.connect(self._reset_admin_settings_to_defaults)

    def _sync_admin_menu_with_config(self):
        """ Synchronisiert die Einstellungen im Admin-Menü mit den aktuellen Werten in der Konfiguration, um sicherzustellen, dass die angezeigten Werte im Admin-Menü mit den tatsächlich verwendeten Einstellungen übereinstimmen. Diese Funktion wird aufgerufen, nachdem die Konfiguration geladen oder aktualisiert wurde, um sicherzustellen, dass alle fehlenden Werte mit den Standardwerten aus der ConfigService ergänzt werden und dass die angezeigten Werte mit den aktuellen Einstellungen übereinstimmen."""
        defaults = self.config_service.get_default_admin_settings()
        moondream = self._get_pipeline_entry("moondream") or {}
        ollama = self._get_pipeline_entry("ollama") or {}
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
            "ollama_enabled": ollama.get("enabled", defaults["ollama_enabled"]),
            "ollama_prompt": ollama.get("prompt", defaults["ollama_prompt"]),
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
        """Wendet die relevanten Einstellungen aus der Konfiguration auf die Laufzeitwerte der GUI an, um sicherzustellen, dass die GUI mit den aktuellen Einstellungen übereinstimmt. Diese Funktion wird aufgerufen, nachdem die Konfiguration geladen oder aktualisiert wurde, um sicherzustellen, dass die Laufzeitwerte der GUI mit den in der Konfiguration festgelegten Werten synchronisiert sind. Es werden die Standardwerte aus der ConfigService verwendet, um fehlende Werte in der Konfiguration zu ergänzen und sicherzustellen, dass alle erforderlichen Werte vorhanden sind, bevor sie auf die Laufzeitwerte angewendet werden."""
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
        """Setzt die Admin-Einstellungen in der Konfiguration auf die Standardwerte zurück, indem die ConfigService verwendet wird, um die Standardwerte zu erhalten und in der aktuellen Konfiguration zu speichern. Nach dem Reset werden die Laufzeitwerte und die Admin-UI sofort neu synchronisiert, damit der Reset direkt sichtbar ist. Es werden auch relevante Timer gestoppt oder gestartet, um sicherzustellen, dass die GUI mit den neuen Einstellungen korrekt funktioniert. Schließlich werden die Pool- und Pipeline-Einstellungen neu geladen und der Zustand des lokalen Ollama-Workers synchronisiert, um sicherzustellen, dass alle Komponenten der GUI mit den zurückgesetzten Einstellungen übereinstimmen."""
        self.config_service.reset_admin_settings(self.config)
        self._save_config()
        # Laufzeitwerte und Admin-UI sofort neu synchronisieren, damit der Reset direkt sichtbar ist.
        self._apply_runtime_settings_from_config()
        self._set_fullscreen(self.config.get("fullscreen", True))
        self._stop_no_person_timer()
        if self.loading_active:
            self._start_pipeline_timeout()
        self._reload_pool_settings()
        self._reload_pipeline_settings()
        self._sync_local_ollama_worker_state()
        self._sync_admin_menu_with_config()

    def _on_photo_delay_changed(self, value):
        """Aktualisiert die Verzögerungszeit für die Fotoaufnahme, indem der übergebene Wert in einen ganzzahligen Wert umgewandelt und auf die Laufzeitvariable angewendet wird. Nach der Aktualisierung wird der neue Wert in der Konfiguration gespeichert, um sicherzustellen, dass die Änderung auch nach einem Neustart der Anwendung erhalten bleibt. Diese Funktion wird aufgerufen, wenn die entsprechende Einstellung im Admin-Menü geändert wird, um die neue Verzögerungszeit sofort wirksam werden zu lassen."""
        self.photo_delay = int(value)
        self._update_config_value("photo_delay", self.photo_delay)

    def _on_close_on_no_person_enabled_changed(self, enabled):
        """
        Aktiviert oder deaktiviert Auto-Close und stoppt bei Bedarf den Warn-Timer.
        :param enabled: True aktiviert Auto-Close, False deaktiviert es.
        """
        self.close_on_no_person_enabled = bool(enabled)
        self._update_config_value("close_on_no_person_enabled", self.close_on_no_person_enabled)
        if not self.close_on_no_person_enabled:
            self._stop_no_person_timer()

    def _on_close_on_no_person_changed(self, value):
        """
        Aktualisiert die Auto-Close-Dauer und setzt den Warnstatus bei Abwesenheit.
        :param value: Neue Dauer in Sekunden.
        """
        self.close_on_no_person_seconds = max(5, min(60, int(value)))
        self._update_config_value("close_on_no_person_seconds", self.close_on_no_person_seconds)
        if self.close_on_no_person_enabled and self._is_open and not self._last_person_present:
            self._missed_presence_checks = 0
            self._update_no_person_warning_state()

    def _on_animation_speed_changed(self, value):
        """
        Aktualisiert die Animationsgeschwindigkeit in der Konfiguration.
        :param value: Neue Geschwindigkeit.
        """
        self.animation_speed = int(value)
        self._update_config_value("animation_speed", self.animation_speed)

    def _on_pipeline_timeout_changed(self, value):
        """
        Aktualisiert das Pipeline-Timeout und startet den Timer bei aktiver Analyse neu.
        :param value: Neue Timeout-Dauer in Sekunden.
        """
        self.pipeline_timeout_seconds = max(1, int(value))
        self._update_config_value("pipeline_timeout_seconds", self.pipeline_timeout_seconds)
        if self.loading_active:
            self._start_pipeline_timeout()

    def _on_face_yolo_confidence_changed(self, value):
        """
        Speichert die neue Konfidenz fuer Face-YOLO in der Config.
        :param value: Neuer Konfidenzwert.
        """
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
        """
        Schaltet Vollbild um und speichert die Einstellung.
        :param enabled: True aktiviert Vollbild.
        """
        self._set_fullscreen(bool(enabled))
        self._update_config_value("fullscreen", bool(enabled))

    def _on_developer_mode_toggled(self, enabled):
        """
        Aktiviert oder deaktiviert den Developer-Mode und aktualisiert die Ansicht.
        :param enabled: True aktiviert den Developer-Mode.
        """
        self.developer_mode = bool(enabled)
        self._update_config_value("developer_mode", self.developer_mode)
        for container in self.active_containers:
            container.set_developer_mode(self.developer_mode)
        # Geschlossene Ansicht sofort aktualisieren, damit der Button korrekt ein-/ausgeblendet wird.
        if not self._is_open and not self.is_animating:
            self.show_closed_folder()

    def _on_pool_enabled_changed(self, enabled):
        """
        Aktiviert oder deaktiviert den Pool.
        :param enabled: True aktiviert den Pool.
        """
        self._update_pool_value("enabled", bool(enabled))

    def _on_pool_max_extra_changed(self, value):
        """
        Setzt die maximale Anzahl zusaetzlicher Personen im Pool.
        :param value: Maximalwert.
        """
        pool_value = max(0, min(3, int(value)))
        self._update_pool_value("max_extra_persons", pool_value)

    def _on_pool_cooldown_changed(self, value):
        """
        Setzt die Pool-Cooldown-Batches.
        :param value: Neue Anzahl Batches.
        """
        self._update_pool_value("cooldown_batches", max(0, int(value)))

    def _on_moondream_enabled(self, enabled):
        """
        Aktiviert oder deaktiviert moondream in der Pipeline.
        :param enabled: True aktiviert moondream.
        """
        self._update_pipeline_value("moondream", "enabled", bool(enabled))
        self._reload_pipeline_settings()

    def _on_moondream_prompt(self, text):
        """
        Aktualisiert den Prompt fuer moondream.
        :param text: Neuer Prompt.
        """
        self._update_pipeline_value("moondream", "prompt", text)

    def _on_ollama_enabled(self, enabled):
        """
        Aktiviert oder deaktiviert ollama und synchronisiert den Worker.
        :param enabled: True aktiviert ollama.
        """
        self._update_pipeline_value("ollama", "enabled", bool(enabled))
        self._reload_pipeline_settings()
        self._sync_local_ollama_worker_state()

    def _on_ollama_prompt(self, text):
        """
        Aktualisiert den Prompt fuer ollama.
        :param text: Neuer Prompt.
        """
        self._update_pipeline_value("ollama", "prompt", text)

    def _on_deepface_enabled(self, enabled):
        """
        Aktiviert oder deaktiviert deepface in der Pipeline.
        :param enabled: True aktiviert deepface.
        """
        self._update_pipeline_value("deepface", "enabled", bool(enabled))
        self._reload_pipeline_settings()

    def _on_deepface_retinaface_changed(self, enabled):
        """
        Setzt die Nutzung von retinaface fuer deepface.
        :param enabled: True nutzt retinaface.
        """
        self._update_pipeline_value("deepface", "use_retinaface", bool(enabled))

    def _on_fer_enabled(self, enabled):
        """
        Aktiviert oder deaktiviert FER in der Pipeline.
        :param enabled: True aktiviert FER.
        """
        self._update_pipeline_value("fer", "enabled", bool(enabled))
        self._reload_pipeline_settings()

    def _on_llm_model_changed(self, value):
        """
        Speichert das gewaehlte LLM-Modell und synchronisiert ollama.
        :param value: Modell-ID.
        """
        self._update_config_value("llm_model", value)
        self._sync_local_ollama_worker_state()

    def _set_fullscreen(self, enabled):
        """
        Setzt den Vollbildstatus der Anwendung.
        :param enabled: True aktiviert Vollbild.
        """
        self.is_fullscreen = bool(enabled)

        # Frameless-Flag dynamisch setzen, damit Fullscreen ohne Fensterrahmen erscheint.
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, self.is_fullscreen)

        if self.is_fullscreen:
            self.showFullScreen()
        else:
            # Beim Verlassen von Fullscreen wieder normales, eingerahmtes Fenster anzeigen.
            self.showNormal()

    def apply_window_state(self):
        """
        Wendet den gespeicherten Fensterzustand an.
        :param: Keine.
        """
        self._set_fullscreen(self.is_fullscreen)

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

    def _on_language_button_clicked(self):
        """
        Verarbeitet den Sprachbutton-Klick und startet den Cooldown.
        :param: Keine.
        """
        # Guard gegen Mehrfachklicks waehrend des Cooldowns
        if self._language_button_cooldown_timer.isActive():
            return
        self.switch_language_logic()
        self._start_language_button_cooldown()

    def _set_language_button_enabled_state(self, enabled):
        """
        Aktiviert/deaktiviert den Sprachbutton und passt die Opacity an.
        :param enabled: True aktiviert den Button.
        """
        btn_language = getattr(self, "btn_language", None)
        if btn_language is None:
            return
        btn_language.setEnabled(bool(enabled))
        # Leicht ausgegraut, solange der Button gesperrt ist
        btn_language.setOpacity(1.0 if enabled else 0.65)

    def _start_language_button_cooldown(self):
        """
        Startet den Cooldown fuer den Sprachbutton.
        :param: Keine.
        """
        self._set_language_button_enabled_state(False)
        self._language_button_cooldown_timer.start(self._language_button_cooldown_ms)

    def _on_language_button_cooldown_timeout(self):
        """
        Beendet den Cooldown und aktiviert den Sprachbutton wieder.
        :param: Keine.
        """
        self._set_language_button_enabled_state(True)

    def switch_language_logic(self):
        """
        Wechselt die UI-Sprache und synchronisiert config.yaml.
        :param: Keine.
        """
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
        """
        Startet den Reset-Countdown oder schliesst die Akte sofort.
        :param: Keine.
        """
        if self._start_reset_countdown():
            return
        self.close_folder(reason="manual")

    def _start_reset_countdown(self):
        """
        Startet den Reset-Countdown.
        :param: Keine.
        :return: True wenn der Countdown aktiv ist oder gestartet wurde, sonst False.
        """
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
        """
        Aktualisiert den Reset-Countdown und fuehrt ggf. den Reset aus.
        :param: Keine.
        """
        self._reset_countdown_remaining -= 1
        if self._reset_countdown_item is not None:
            self._reset_countdown_item.set_remaining(self._reset_countdown_remaining)
        if self._reset_countdown_remaining <= 0:
            self._reset_countdown_timer.stop()
            self._clear_reset_countdown()
            self.close_folder(reason="manual_countdown")

    def _clear_reset_countdown(self):
        """
        Entfernt die Countdown-Anzeige und stellt den Reset-Button wieder her.
        :param: Keine.
        """
        if self._reset_countdown_item is not None:
            self.scene.removeItem(self._reset_countdown_item)
            self._reset_countdown_item = None
        self._set_reset_button_empty(False)

    def _set_reset_button_empty(self, is_empty):
        """
        Tauscht das Reset-Button-Pixmap gegen die leere Variante.
        :param is_empty: True zeigt die leere Variante.
        """
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
        """
        Reagiert auf Tastaturevents und oeffnet das Admin-Menue mit E.
        :param event: Tastaturevent.
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
        Passt die View an und aktualisiert die Admin-Menue-Geometrie.
        :param event: Resize-Event.
        """
        super().resizeEvent(event)
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        if self.admin_menu.isVisible():
            self.admin_menu.update_geometry(self.size())

