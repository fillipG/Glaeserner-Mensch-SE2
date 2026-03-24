"""
gui/mixins/config_handlers_mixin.py
-------------------------------------
Mixin für Config-Verwaltung und Admin-Menü-Handler der ScalingAkteGUI.

Zuständigkeiten:
- config.yaml laden und speichern (_load_config, _save_config)
- Laufzeitwerte aus Config anwenden (_apply_runtime_settings_from_config)
- Admin-Menü mit aktuellen Werten synchronisieren (_sync_admin_menu_with_config)
- Alle Admin-Menü-Signale mit Callbacks verknüpfen (_connect_admin_menu)
- 30+ Handler für einzelne Einstellungs-Änderungen (_on_*_changed)
- Pipeline/Pool/Ollama-Worker neu laden wenn nötig

Hinweis zum Reload-Mechanismus:
    Einstellungsänderungen im Admin-Menü rufen immer _save_config() auf.
    Änderungen die Pipeline-Verhalten betreffen, rufen zusätzlich
    _reload_pipeline_settings() auf (leitet an PipelineWorker weiter).
    Ollama-relevante Änderungen rufen _sync_local_ollama_worker_state() auf.

Benötigte self-Attribute (in ScalingAkteGUI.__init__ gesetzt):
    config, config_service, admin_menu, developer_mode, is_fullscreen
    photo_delay, animation_speed, close_on_no_person_enabled, etc.
    _pipeline (PipelineWorker), _local_worker_manager (LocalWorkerManager)
"""

from constants import PipelineStage
from PyQt6.QtWidgets import QMessageBox


class ConfigHandlersMixin:
    """Mixin: Config-Verwaltung und Admin-Menü-Handler."""

    # =========================================================
    # Config laden / speichern
    # =========================================================

    def _load_config(self):
        """Lädt die gesamte config.yaml und setzt fehlende Defaults."""
        return self.config_service.load()

    def _save_config(self):
        """Speichert die aktuelle Konfiguration zurück in die config.yaml."""
        self.config_service.save(self.config)

    def _ensure_config_defaults(self, config):
        """Stellt sicher, dass alle erforderlichen Standardwerte vorhanden sind."""
        self.config_service.ensure_defaults(config)

    def _apply_runtime_settings_from_config(self):
        """
        Überträgt Werte aus der geladenen Config auf die Laufzeit-Attribute.
        Wird beim Start und nach einem Admin-Reset aufgerufen.
        Ohne diesen Schritt würden Config-Änderungen erst nach Neustart wirken.
        """
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
        self._live_deepface_svc.apply_config(self.config)

    # =========================================================
    # Pipeline-Einträge
    # =========================================================

    def _get_pipeline_entry(self, model_id):
        """
        Sucht in der Config nach einem Pipeline-Eintrag anhand seiner ID.
        :param model_id: ID des gesuchten Modells (z.B. PipelineStage.DEEPFACE).
        :return: Dict mit Modell-Config oder None.
        """
        pipeline = self.config.setdefault("pipeline", [])
        return next((p for p in pipeline if p.get("id") == model_id), None)

    def _update_config_value(self, key, value):
        """Aktualisiert einen Top-Level-Wert in der Config und speichert."""
        self.config[key] = value
        self._save_config()

    def _get_face_yolo_confidence(self):
        """
        Liest den Konfidenzwert für Face-YOLO aus der Config.
        Unterstützt sowohl neue Struktur (face_yolo.confidence) als auch
        den alten Key (face_yolo_confidence) für Rückwärtskompatibilität.
        """
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
        """
        Aktualisiert einen Wert im Pipeline-Eintrag eines Modells und speichert.
        Falls der Eintrag nicht existiert, werden Defaults ergänzt und erneut gesucht.
        """
        entry = self._get_pipeline_entry(model_id)
        if entry is None:
            self.config_service.ensure_defaults(self.config)
            entry = self._get_pipeline_entry(model_id)
        if entry is None:
            return
        entry[key] = value
        self._save_config()

    def _update_pool_value(self, key, value):
        """
        Aktualisiert einen Wert in der Pool-Config und lädt Pool-Einstellungen neu.
        Der sofortige Reload stellt sicher, dass Änderungen im laufenden Betrieb wirken.
        """
        pool = self.config.setdefault("pool", {})
        if not isinstance(pool, dict):
            pool = {}
            self.config["pool"] = pool
        pool[key] = value
        self._save_config()
        self._reload_pool_settings()

    # =========================================================
    # Reload-Signale an Worker
    # =========================================================

    def _reload_pool_settings(self):
        """Signalisiert dem PipelineWorker, den Pool neu zu laden."""
        pipeline = getattr(self, "_pipeline", None)
        if pipeline is not None and hasattr(pipeline, "request_pool_reload"):
            pipeline.request_pool_reload()
        pool_loader = getattr(self, "_pool_loader", None)
        if pool_loader is not None and hasattr(pool_loader, "reload"):
            pool_loader.reload(config_data=self.config)

    def _reload_pipeline_settings(self):
        """Signalisiert dem PipelineWorker, die Pipeline-Config neu zu laden."""
        pipeline = getattr(self, "_pipeline", None)
        if pipeline is not None and hasattr(pipeline, "request_pipeline_reload"):
            pipeline.request_pipeline_reload()

    def _sync_local_ollama_worker_state(self):
        """
        Synchronisiert den lokalen Ollama-Worker nach Einstellungsänderungen.
        Z.B. wenn Ollama deaktiviert/aktiviert oder das Modell gewechselt wurde.
        """
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

    # =========================================================
    # Admin-Menü verdrahten
    # =========================================================

    def _connect_admin_menu(self):
        """
        Verbindet alle Admin-Menü-Signale mit den entsprechenden Handler-Methoden.
        Wird einmalig im __init__ aufgerufen.
        """
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
        self.admin_menu.live_deepface_enabled_changed.connect(self._on_live_deepface_enabled_changed)
        self.admin_menu.live_deepface_interval_changed.connect(self._on_live_deepface_interval_changed)
        self.admin_menu.sounds_enabled_changed.connect(self._on_sounds_enabled_changed)
        self.admin_menu.llm_model_changed.connect(self._on_llm_model_changed)
        self.admin_menu.reset_defaults_requested.connect(self._reset_admin_settings_to_defaults)
        self.admin_menu.statistics_enabled_changed.connect(self._on_statistics_enabled_changed)
        self.admin_menu.statistics_reset_requested.connect(self._on_statistics_reset_requested)

    def _sync_admin_menu_with_config(self):
        """
        Aktualisiert die Anzeige im Admin-Menü mit den aktuellen Config-Werten.
        Wird aufgerufen beim Öffnen des Admin-Menüs und nach einem Reset.
        """
        defaults = self.config_service.get_default_admin_settings()
        moondream = self._get_pipeline_entry(PipelineStage.MOONDREAM) or {}
        ollama    = self._get_pipeline_entry(PipelineStage.OLLAMA) or {}
        deepface  = self._get_pipeline_entry(PipelineStage.DEEPFACE) or {}
        fer       = self._get_pipeline_entry(PipelineStage.FER) or {}
        pool      = self.config.get("pool", {})
        settings = {
            "photo_delay": self.config.get("photo_delay", defaults["photo_delay"]),
            "sounds_enabled": self.config.get("sounds", {}).get("enabled", defaults["sounds_enabled"]),
            "close_on_no_person_enabled": self.config.get(
                "close_on_no_person_enabled", defaults["close_on_no_person_enabled"]
            ),
            "close_on_no_person_seconds": self.config.get(
                "close_on_no_person_seconds", defaults["close_on_no_person_seconds"]
            ),
            "animation_speed": self.config.get("animation_speed", defaults["animation_speed"]),
            "pipeline_timeout_seconds": self.config.get(
                "pipeline_timeout_seconds", defaults["pipeline_timeout_seconds"]
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
            "deepface_use_retinaface": deepface.get("use_retinaface", defaults["deepface_use_retinaface"]),
            "fer_enabled": fer.get("enabled", defaults["fer_enabled"]),
            "live_deepface_enabled": self.config.get("live_deepface", {}).get(
                "enabled", defaults["live_deepface_enabled"]
            ),
            "live_deepface_interval_seconds": self.config.get("live_deepface", {}).get(
                "interval_seconds", defaults["live_deepface_interval_seconds"]
            ),
            "llm_model": self.config.get("llm_model", defaults["llm_model"]),
            "statistics_enabled": self.config.get("statistics", {}).get("enabled", defaults["statistics_enabled"]),
        }
        self.admin_menu.apply_settings(settings)
        # Statistik-Anzeige aktualisieren (alle 4 Zeiträume immer sichtbar)
        self.admin_menu.refresh_stats(
            self._stats_svc.get_today(),
            self._stats_svc.get_period("daily"),
            self._stats_svc.get_month(),
            self._stats_svc.get_total(),
        )

    def _reset_admin_settings_to_defaults(self):
        """
        Setzt alle Admin-Einstellungen auf die Standardwerte zurück.
        Danach werden Laufzeitwerte, UI und alle Worker sofort synchronisiert.
        """
        self.config_service.reset_admin_settings(self.config)
        self._save_config()
        # Alle Laufzeitwerte sofort aktualisieren, damit der Reset direkt sichtbar ist
        self._apply_runtime_settings_from_config()
        # StatisticsService-Laufzeitstatus nach Reset synchronisieren
        self._stats_svc.enabled = bool(self.config.get("statistics", {}).get("enabled", True))
        self._set_fullscreen(self.config.get("fullscreen", True))
        self._stop_no_person_timer()
        if self.loading_active:
            self._start_pipeline_timeout()
        self._reload_pool_settings()
        self._reload_pipeline_settings()
        self._sync_local_ollama_worker_state()
        self._sync_admin_menu_with_config()

    # =========================================================
    # Admin-Menü Handler (ein Handler pro Einstellung)
    # =========================================================

    def _on_photo_delay_changed(self, value):
        """Setzt die Foto-Countdown-Dauer in Sekunden."""
        self.photo_delay = int(value)
        self._update_config_value("photo_delay", self.photo_delay)

    def _on_close_on_no_person_enabled_changed(self, enabled):
        """
        Aktiviert/deaktiviert Auto-Close.
        Bei Deaktivierung wird der Warn-Timer sofort gestoppt.
        """
        self.close_on_no_person_enabled = bool(enabled)
        self._update_config_value("close_on_no_person_enabled", self.close_on_no_person_enabled)
        if not self.close_on_no_person_enabled:
            self._stop_no_person_timer()

    def _on_close_on_no_person_changed(self, value):
        """
        Setzt die Auto-Close-Dauer und setzt den Warn-Zustand zurück.
        Bei aktiver Abwesenheits-Überwachung wird der Zähler zurückgesetzt.
        """
        self.close_on_no_person_seconds = max(5, min(60, int(value)))
        self._update_config_value("close_on_no_person_seconds", self.close_on_no_person_seconds)
        if self.close_on_no_person_enabled and self._is_open and not self._last_person_present:
            self._missed_presence_checks = 0
            self._update_no_person_warning_state()

    def _on_animation_speed_changed(self, value):
        """Setzt die ms pro Frame für Ordner-Animationen."""
        self.animation_speed = int(value)
        self._update_config_value("animation_speed", self.animation_speed)

    def _on_pipeline_timeout_changed(self, value):
        """
        Setzt das Pipeline-Timeout und startet den Timer neu wenn Analyse läuft.
        """
        self.pipeline_timeout_seconds = max(1, int(value))
        self._update_config_value("pipeline_timeout_seconds", self.pipeline_timeout_seconds)
        if self.loading_active:
            self._start_pipeline_timeout()

    def _on_face_yolo_confidence_changed(self, value):
        """
        Setzt den Konfidenzwert für Face-YOLO und bereinigt den veralteten Key.
        """
        self.face_yolo_confidence = max(0.10, min(0.90, float(value)))
        face_yolo_cfg = self.config.setdefault("face_yolo", {})
        if not isinstance(face_yolo_cfg, dict):
            face_yolo_cfg = {}
            self.config["face_yolo"] = face_yolo_cfg
        face_yolo_cfg["confidence"] = round(self.face_yolo_confidence, 2)
        # Veralteten Top-Level-Key entfernen falls vorhanden
        if "face_yolo_confidence" in self.config:
            del self.config["face_yolo_confidence"]
        self._save_config()

    def _on_fullscreen_toggled(self, enabled):
        """Schaltet Vollbild ein/aus und speichert die Einstellung."""
        self._set_fullscreen(bool(enabled))
        self._update_config_value("fullscreen", bool(enabled))

    def _on_developer_mode_toggled(self, enabled):
        """
        Aktiviert/deaktiviert den Developer-Mode.
        Aktualisiert alle Container und die geschlossene Ansicht sofort.
        """
        self.developer_mode = bool(enabled)
        self._update_config_value("developer_mode", self.developer_mode)
        for container in self.active_containers:
            container.set_developer_mode(self.developer_mode)
        # Geschlossene Ansicht neu aufbauen damit der Entwickler-Button erscheint/verschwindet
        if not self._is_open and not self.is_animating:
            self.show_closed_folder()

    def _on_pool_enabled_changed(self, enabled):
        """Aktiviert/deaktiviert die Pool-Funktion."""
        self._update_pool_value("enabled", bool(enabled))

    def _on_pool_max_extra_changed(self, value):
        """Setzt die maximale Anzahl zusätzlicher Pool-Personen (0-3)."""
        self._update_pool_value("max_extra_persons", max(0, min(3, int(value))))

    def _on_pool_cooldown_changed(self, value):
        """Setzt die Anzahl Batches, die eine Pool-Person pausiert."""
        self._update_pool_value("cooldown_batches", max(0, int(value)))

    def _on_moondream_enabled(self, enabled):
        """
        Aktiviert/deaktiviert Moondream.
        Moondream bleibt intern immer aktiviert (wird als Textquelle benötigt),
        deshalb wird hier immer True gesetzt.
        """
        self._update_pipeline_value(PipelineStage.MOONDREAM, "enabled", True)
        self._reload_pipeline_settings()

    def _on_moondream_prompt(self, text):
        """Aktualisiert den Beschreibungs-Prompt für Moondream."""
        self._update_pipeline_value(PipelineStage.MOONDREAM, "prompt", text)

    def _on_ollama_enabled(self, enabled):
        """
        Aktiviert/deaktiviert Ollama und synchronisiert den Worker-Prozess.
        Wenn Ollama deaktiviert wird, wird Moondream als Textquelle genutzt.
        """
        self._update_pipeline_value(PipelineStage.OLLAMA, "enabled", bool(enabled))
        self._reload_pipeline_settings()
        self._sync_local_ollama_worker_state()

    def _on_ollama_prompt(self, text):
        """Aktualisiert den Kriminalgeschichten-Prompt für Ollama."""
        self._update_pipeline_value(PipelineStage.OLLAMA, "prompt", text)

    def _on_deepface_enabled(self, enabled):
        """Aktiviert/deaktiviert DeepFace (Emotionserkennung)."""
        self._update_pipeline_value(PipelineStage.DEEPFACE, "enabled", bool(enabled))
        self._reload_pipeline_settings()

    def _on_deepface_retinaface_changed(self, enabled):
        """Schaltet RetinaFace als Gesichts-Detektor für DeepFace ein/aus."""
        self._update_pipeline_value(PipelineStage.DEEPFACE, "use_retinaface", bool(enabled))

    def _on_fer_enabled(self, enabled):
        """Aktiviert/deaktiviert FER (Fallback-Emotionserkennung)."""
        self._update_pipeline_value(PipelineStage.FER, "enabled", bool(enabled))
        self._reload_pipeline_settings()

    def _on_live_deepface_enabled_changed(self, enabled):
        """Aktiviert/deaktiviert die Live-Analyse im Kamera-Preview."""
        self._live_deepface_svc.set_enabled(enabled, save_callback=self._save_config)

    def _on_live_deepface_interval_changed(self, value):
        """Setzt das Intervall für die Live-Analyse in Sekunden."""
        self._live_deepface_svc.set_interval(value, save_callback=self._save_config)

    def _on_sounds_enabled_changed(self, enabled: bool):
        """Aktiviert/deaktiviert alle Sounds und speichert die Einstellung."""
        self._sound_svc.set_enabled(bool(enabled))
        sounds_cfg = self.config.setdefault("sounds", {})
        if not isinstance(sounds_cfg, dict):
            sounds_cfg = {}
            self.config["sounds"] = sounds_cfg
        sounds_cfg["enabled"] = bool(enabled)
        self._save_config()

    def _on_llm_model_changed(self, value):
        """
        Wechselt das LLM-Modell und synchronisiert den Ollama-Worker.
        Der Worker lädt das neue Modell bei der nächsten Anfrage.
        """
        self._update_config_value("llm_model", value)
        self._sync_local_ollama_worker_state()

    def _on_statistics_enabled_changed(self, enabled):
        """
        Schaltet die Besucherstatistik-Aufzeichnung ein oder aus.
        :param enabled: True aktiviert die Statistik.
        """
        statistics = self.config.setdefault("statistics", {})
        statistics["enabled"] = enabled
        self._stats_svc.enabled = enabled
        self._save_config()
        self.admin_menu.refresh_stats(
            self._stats_svc.get_today(),
            self._stats_svc.get_period("daily"),
            self._stats_svc.get_month(),
            self._stats_svc.get_total(),
        )

    def _on_statistics_reset_requested(self):
        """
        Löscht alle Besucherstatistik-Daten nach Bestätigung.
        Sicherheitsabfrage verhindert versehentliches Löschen im Museumsbetrieb.
        """
        reply = QMessageBox.question(
            self,
            "Statistik zurücksetzen",
            "Alle Besucherstatistik-Daten unwiderruflich löschen?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._stats_svc.reset()
        self.admin_menu.refresh_stats(
            self._stats_svc.get_today(),
            self._stats_svc.get_period("daily"),
            self._stats_svc.get_month(),
            self._stats_svc.get_total(),
        )
