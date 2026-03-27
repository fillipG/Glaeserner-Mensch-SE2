"""
gui/mixins/animation_mixin.py
------------------------------
Mixin für alle Animations- und Ordner-Anzeige-Methoden der ScalingAkteGUI.

Zuständigkeiten:
- Öffnungs- und Schließ-Animation der Mappe (Video-Playback via OpenCV)
- Flip-Animation beim Wechsel zwischen Personen-Batches
- Anzeige des geschlossenen Ordners (inkl. Logos, Kamera-Preview-Platzhalter)
- Anzeige des offenen Ordners (inkl. PersonContainer, Buttons)
- Foto-Countdown (CircularTimerItem vor dem Öffnen)

Benötigte self-Attribute (in ScalingAkteGUI.__init__ gesetzt):
    scene, video_cap, video_item, is_animating, _is_open
    _animation_end_callback, _pending_close, _clear_pipeline_outputs_on_close
    _sound_svc, wait_timer, wait_timer_item, developer_mode, animation_speed
    photo_delay, PATHS, SCENE_WIDTH, SCENE_HEIGHT
"""

import os
import shutil
import cv2

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QFont, QImage, QPen, QColor
from PyQt6.QtWidgets import QLabel, QPushButton

from ..gui_constants import SCENE_WIDTH, SCENE_HEIGHT, PATHS
from ..gui_state import GUIState
from ..ui_widgets import CircularTimerItem


class AnimationMixin:
    """Mixin: Animations- und Ordner-Anzeige-Logik."""

    # =========================================================
    # Öffnen / Schließen der Mappe
    # =========================================================

    def close_folder(self, reason: str = "", animated: bool = True):
        """
        Schließt die Akte mit optionaler Schließ-Animation.

        Je nach Grund wird entschieden, ob die Pipeline-Ausgaben danach
        bereinigt werden (z.B. bei manuellem Reset, aber nicht bei Fehler).

        :param reason: Grund des Schließens. Werte:
            "manual"           – Nutzer hat Reset gedrückt (sofort)
            "manual_countdown" – Nutzer hat Reset gedrückt (Countdown)
            "auto_close"       – Person hat den Bereich verlassen
            "pipeline_timeout" – KI hat zu lange gebraucht
            "empty_result"     – Keine Person erkannt
        :param animated: True = Schließ-Video abspielen, False = sofort schließen.
        """
        # Kamera frühzeitig im Worker vorwärmen, damit nach der Schließ-Animation
        # schneller wieder ein Livebild statt eines langen schwarzen Platzhalters erscheint.
        self.camera_prewarm_requested.emit()

        # Nur bei bewusstem Nutzer- oder System-Reset werden alte Dateien bereinigt.
        # Bei Fehlern (z.B. pipeline_timeout) ist die Bereinigung ebenfalls erwünscht,
        # damit der nächste Durchlauf sauber startet.
        self._clear_pipeline_outputs_on_close = reason in {
            "manual",
            "manual_countdown",
            "auto_close",
            "pipeline_timeout",
            "empty_result",
        }
        if self.is_animating:
            # Animation läuft bereits → Schließen nach Ende der aktuellen Animation
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
        Löscht alle Dateien in den Pipeline-Ausgabeverzeichnissen.

        Wird beim Schließen der Mappe aufgerufen, damit der nächste Durchlauf
        nicht auf Restdateien aus dem vorherigen Batch trifft.
        Nur aktiv wenn _clear_pipeline_outputs_on_close gesetzt ist.
        """
        cleanup_dirs = [
            PATHS["final_dir"],
            PATHS["ollama_inbox"],
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

    # =========================================================
    # Geschlossener Ordner
    # =========================================================

    def show_closed_folder(self):
        """
        Zeigt den geschlossenen Ordner an.

        Bereinigt die Szene, lädt das Hintergrundbild, positioniert Logos,
        erstellt den Kamera-Preview-Bereich und setzt alle Zustandsvariablen
        für den nächsten Durchlauf zurück.
        """
        should_play_close_sound = self._is_open
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

        if should_play_close_sound:
            self._sound_svc.play("folder_close")
        if self.wait_timer.isActive():
            self.wait_timer.stop()

        # Hintergrundbild des geschlossenen Ordners
        path = PATHS["closed_folder"]
        if os.path.exists(path):
            self.scene.addPixmap(
                QPixmap(path).scaled(
                    SCENE_WIDTH, SCENE_HEIGHT,
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding
                )
            )

        # Statische Logos (Autor: Lukas Käuper)
        logo_configs = [
            {"path": PATHS.get("logo_bmftr"), "scale": 0.4,  "pos": (30, SCENE_HEIGHT - 30)},
            {"path": PATHS.get("logo_ki_owl"), "scale": 0.11, "pos": (400, SCENE_HEIGHT - 30)},
            {"path": PATHS.get("logo_th_owl"), "scale": 0.5,  "pos": (30, 200)},
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

        # Kamera-Vorschaubereich (Autor: Lukas Käuper)
        self._cam_display_w = 896   # Breite in Pixeln
        self._cam_display_h = 504   # Höhe in Pixeln
        cam_x = 20
        cam_y = ((SCENE_HEIGHT - self._cam_display_h) // 2) - 30

        # Umrahmung des Kamerabildes
        border = self.scene.addRect(cam_x - 3, cam_y - 3, self._cam_display_w + 6, self._cam_display_h + 6)
        border.setPen(QPen(QColor("#f4e4bc"), 3))
        border.setZValue(9)

        # Schwarzer Platzhalter bis das erste Kamerabild kommt
        placeholder = QPixmap(self._cam_display_w, self._cam_display_h)
        placeholder.fill(QColor("black"))
        new_item = self.scene.addPixmap(placeholder)
        new_item.setPos(cam_x, cam_y)
        new_item.setZValue(10)
        # Zuweisung via singleShot vermeidet Timing-Probleme beim Szene-Aufbau
        QTimer.singleShot(0, lambda: setattr(self, "camera_pixmap_item", new_item))

        cam_label = QLabel("LIVE KAMERA")
        cam_label.setFont(QFont("Graduate", 14, QFont.Weight.Bold))
        cam_label.setStyleSheet("color: #f4e4bc; background: transparent;")
        cam_label_proxy = self.scene.addWidget(cam_label)
        cam_label_proxy.setPos(cam_x, cam_y - 35)
        cam_label_proxy.setZValue(11)

        # Foto-Countdown-Anzeige (wird erst nach Personenerkennung sichtbar)
        self.wait_timer_item = CircularTimerItem(self.photo_delay, diameter=240)
        self.scene.addItem(self.wait_timer_item)
        self.wait_timer_item.hide()
        self.wait_timer_item.setPos(
            SCENE_WIDTH - self.wait_timer_item.diameter - 450,
            SCENE_HEIGHT - self.wait_timer_item.diameter - 120,
        )

        # Entwickler-Button zum manuellen Öffnen der Mappe
        self.btn_open = None
        if self.developer_mode:
            self.btn_open = QPushButton("Mappe öffnen")
            self.btn_open.setFixedSize(300, 80)
            self.btn_open.setStyleSheet(
                "QPushButton { background-color: #3d2b1f; color: #f4e4bc; "
                "border: 3px solid #f4e4bc; border-radius: 15px; "
                "font-family: 'Graduate'; font-size: 24px; font-weight: bold; } "
                "QPushButton:hover { background-color: #5a4030; }"
            )
            self.btn_open.clicked.connect(self.show_animation_with_timer)
            proxy = self.scene.addWidget(self.btn_open)
            proxy.setPos(550, 100)

        # Signal an YOLOWorker: Kamera wieder starten
        self.folder_closed.emit()

    # =========================================================
    # Offener Ordner
    # =========================================================

    def show_open_folder(self):
        """
        Zeigt die offene Mappe mit den Personen-Containern an.

        Wird nach Abschluss der Öffnungs-Animation aufgerufen.
        Aktiviert ggf. das Auto-Close-Monitoring.
        """
        self._is_open = True
        self._sound_svc.play("folder_open")
        self._set_state(GUIState.RESULTS_READY)
        self._stop_no_person_timer()

        # Auto-Close-Monitoring aktivieren, wenn Person die Mappe ausgelöst hat
        self._auto_close_monitoring_enabled = bool(self._auto_close_monitoring_pending)
        self._auto_close_monitoring_pending = False
        if self._auto_close_monitoring_enabled and self.close_on_no_person_enabled:
            print("Auto-close monitoring active")
            self._set_state(GUIState.PRESENCE_MONITORING)
            self.presence_monitoring_requested.emit()

        self.scene.clear()
        bg = PATHS["open_folder"]
        if os.path.exists(bg):
            self.scene.addPixmap(
                QPixmap(bg).scaled(
                    SCENE_WIDTH, SCENE_HEIGHT,
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        self.setup_ui_elements()
        self.setup_buttons()
        self.update_descriptions_from_files()

        for container in self.active_containers:
            container.trigger_typing()

        # Falls die Person schon weg ist, direkt den Warnstatus prüfen
        if (
            self._auto_close_monitoring_enabled
            and self.close_on_no_person_enabled
            and not self._last_person_present
        ):
            self._missed_presence_checks = 0
            self._update_no_person_warning_state()

    def show_flip_video(self):
        """
        Spielt das Umblätter-Video ab und wechselt danach zur offenen Mappe.
        Wird verwendet, wenn ein neuer Batch kommt während die Mappe schon offen ist.
        """
        self._set_state(GUIState.FLIPPING)
        self.start_animation(PATHS["flip_animation"], end_callback=self.show_open_folder)

    # =========================================================
    # Foto-Countdown vor dem Öffnen
    # =========================================================

    def show_animation_with_timer(self):
        """
        Entscheidet, ob die Öffnungs-Animation sofort oder nach Countdown starten soll.
        Wird aufgerufen, wenn YOLO ein Foto auslöst (oder manuell per Entwickler-Button).
        """
        if self.is_animating or self.active_containers:
            return
        if self.wait_timer_item is None:
            self.start_animation()
            return
        try:
            self.wait_timer_item.isVisible()
        except RuntimeError:
            # Qt-Objekt wurde bereits zerstört
            self.wait_timer_item = None
            self.start_animation()
            return
        self._start_wait_timer()

    def _start_wait_timer(self):
        """
        Startet den Foto-Countdown (CircularTimerItem).
        Im Developer-Mode wird der Countdown übersprungen.
        """
        if self.developer_mode:
            self.start_animation()
            return
        duration = max(1, int(self.photo_delay))
        self._wait_duration_s = duration
        self._wait_start_time = __import__("time").perf_counter()
        if self.wait_timer_item is None:
            self.start_animation()
            return
        self.wait_timer_item.set_progress(0.0, duration)
        self.wait_timer_item.show()
        btn_open = getattr(self, "btn_open", None)
        if btn_open is not None:
            btn_open.setEnabled(False)
        # 33ms = ~30 FPS für flüssige Countdown-Animation
        self.wait_timer.start(33)

    def _update_wait_timer(self):
        """
        Wird vom wait_timer (33ms-Takt) aufgerufen.
        Aktualisiert den Countdown und startet bei 0 die Animation.
        """
        if self._wait_start_time is None:
            return
        elapsed = __import__("time").perf_counter() - self._wait_start_time
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

    # =========================================================
    # Video-Playback (Öffnen / Schließen / Flip)
    # =========================================================

    def start_animation(self, checked=False, video_path=PATHS["open_animation"], end_callback=None):
        """
        Startet ein Ordner-Animations-Video.

        :param checked: Wird als video_path interpretiert wenn es ein String ist
                        (Qt-Signal-Kompatibilität: clicked(bool) kann str übergeben).
        :param video_path: Pfad zur Videodatei.
        :param end_callback: Funktion die nach dem Video aufgerufen wird.
                             Standard: show_open_folder.
        """
        # Qt-clicked-Signal übergibt manchmal einen String als erstes Argument
        if isinstance(checked, (str, os.PathLike)):
            video_path = checked
            checked = False

        # State setzen je nach End-Ziel
        if end_callback == self.show_open_folder:
            if self._is_open:
                self._set_state(GUIState.FLIPPING)
            else:
                self._set_state(GUIState.OPENING)

        self.hide_loading_indicator()

        # Wenn die Videodatei fehlt, direkt zum End-Callback springen
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
        Liest das nächste Video-Frame und zeigt es in der Szene an.
        Wird rekursiv via QTimer aufgerufen (Intervall = animation_speed ms).
        Bei Videoende: End-Callback aufrufen + ggf. ausstehenden Close ausführen.
        """
        if not self.is_animating or self.video_item is None or self.video_cap is None:
            return
        ret, frame = self.video_cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            q_img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            self.video_item.setPixmap(QPixmap.fromImage(q_img).scaled(SCENE_WIDTH, SCENE_HEIGHT))
            # Nächstes Frame nach animation_speed ms anfordern
            QTimer.singleShot(self.animation_speed, self.update_video_frame)
        else:
            # Video fertig: aufräumen und End-Callback auslösen
            self.is_animating = False
            self.video_cap.release()
            self.video_cap = None
            self.video_item = None
            end_callback = self._animation_end_callback
            QTimer.singleShot(100, end_callback)

            # Ausstehender Schließ-Aufruf (während Animation angefragt) jetzt ausführen
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
