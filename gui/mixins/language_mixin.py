"""
gui/mixins/language_mixin.py
-----------------------------
Mixin für Sprachwechsel-Logik der ScalingAkteGUI.

Zuständigkeiten:
- switch_language_logic: Wechselt zwischen Deutsch und Englisch
- Speichert Spracheinstellung in config.yaml
- Aktualisiert alle aktiven PersonContainer
- Übersetzt angezeigte Beschreibungen neu
- Cooldown-Mechanismus verhindert Mehrfachklicks am Sprach-Button

Benötigte self-Attribute (in ScalingAkteGUI.__init__ gesetzt):
    current_language, translator, active_containers
    _language_button_cooldown_timer, _language_button_cooldown_ms
    btn_language (nur wenn Mappe offen)
"""

from service import TranslationService


class LanguageMixin:
    """Mixin: Sprach-Umschaltung und Container-Update."""

    # =========================================================
    # Config-Zugriff (delegiert an ConfigHandlersMixin)
    # =========================================================

    def _load_language_from_config(self):
        """Liest die aktuelle Sprache aus der Konfiguration. Standard: 'de'."""
        return self.config.get("language", "de")

    def _save_language_to_config(self, language):
        """Schreibt die neue Sprache in die Konfiguration und speichert."""
        self._update_config_value("language", language)

    def _apply_language_to_containers(self, language):
        """Setzt die Anzeigesprache für alle aktiven PersonContainer."""
        for container in self.active_containers:
            container.apply_language(language)

    def _refresh_descriptions_for_language(self):
        """
        Übersetzt alle angezeigten Beschreibungen nach einem Sprachwechsel neu.
        Wird aufgerufen nachdem self.translator auf die neue Sprache gesetzt wurde.
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

    # =========================================================
    # Button-Interaktion
    # =========================================================

    def _on_language_button_clicked(self):
        """
        Verarbeitet den Klick auf den Sprachbutton.
        Der Cooldown-Timer verhindert, dass schnelle Mehrfachklicks
        zu mehrfachen Übersetzungs-Anfragen führen.
        """
        # Guard: Während Cooldown aktiv ist, Klick ignorieren
        if self._language_button_cooldown_timer.isActive():
            return
        self.switch_language_logic()
        self._start_language_button_cooldown()

    def _set_language_button_enabled_state(self, enabled):
        """
        Aktiviert/deaktiviert den Sprachbutton und passt die Deckkraft an.
        Deaktivierter Button wird halbtransparent dargestellt.
        :param enabled: True = aktiv, False = gesperrt (Cooldown).
        """
        btn_language = getattr(self, "btn_language", None)
        if btn_language is None:
            return
        btn_language.setEnabled(bool(enabled))
        # Visuelle Rückmeldung: Leicht ausgegraut während Cooldown
        btn_language.setOpacity(1.0 if enabled else 0.65)

    def _start_language_button_cooldown(self):
        """
        Startet den Cooldown nach einem Sprachwechsel.
        Während des Cooldowns ist der Sprachbutton deaktiviert (5 Sekunden Standard).
        """
        self._set_language_button_enabled_state(False)
        self._language_button_cooldown_timer.start(self._language_button_cooldown_ms)

    def _on_language_button_cooldown_timeout(self):
        """Callback: Cooldown beendet → Sprachbutton wieder aktivieren."""
        self._set_language_button_enabled_state(True)

    # =========================================================
    # Sprachwechsel-Logik
    # =========================================================

    def switch_language_logic(self):
        """
        Führt den Sprachwechsel durch.

        Wechselt zwischen "de" (Deutsch) und "en" (Englisch):
        - Aktualisiert self.current_language
        - Erstellt neuen TranslationService für die neue Sprache
        - Aktualisiert alle Container-Labels
        - Übersetzt angezeigte Beschreibungen neu
        - Speichert die neue Sprache in config.yaml
        """
        btn_language = getattr(self, "btn_language", None)
        if btn_language is None:
            return
        if btn_language.is_toggled:
            self.current_language = "en"
            self._apply_language_to_containers("en")
            self._save_language_to_config("en")
            print("Status: Englisch")
        else:
            self.current_language = "de"
            self._apply_language_to_containers("de")
            self._save_language_to_config("de")
            print("Status: Deutsch")

        # Neuen TranslationService für die gewählte Sprache erstellen
        self.translator = TranslationService(target_lang=self.current_language)
        # Alle angezeigten Beschreibungen neu übersetzen
        self._refresh_descriptions_for_language()
