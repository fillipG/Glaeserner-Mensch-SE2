"""
gui/mixins/presence_mixin.py
-----------------------------
Mixin für Anwesenheits-Überwachung und Auto-Close der ScalingAkteGUI.

Zuständigkeiten:
- on_person_presence_changed: Reagiert auf Anwesenheits-Signale des YOLOWorkers
- Auto-Close: Zählt verpasste Checks und schließt die Mappe automatisch
- Warnung: Lässt den Reset-Button blinken wenn Auto-Close nahe ist

Ablauf des Auto-Close:
    1. YOLOWorker sendet person_presence_changed(False)
    2. _missed_presence_checks wird erhöht
    3. Bei _get_warning_start_missed_checks() → Reset-Button blinkt
    4. Bei _get_auto_close_missed_check_limit() → close_folder("auto_close")

Benötigte self-Attribute (in ScalingAkteGUI.__init__ gesetzt):
    _last_person_present, _missed_presence_checks
    _no_person_warning_timer, _is_open, is_animating, loading_active
    _auto_close_monitoring_enabled, close_on_no_person_enabled
    close_on_no_person_seconds, no_person_check_interval_ms
"""

from PyQt6.QtCore import pyqtSlot


class PresenceMixin:
    """Mixin: Anwesenheits-Überwachung und Auto-Close."""

    def _stop_no_person_timer(self):
        """
        Stoppt den Warn-Timer und setzt den Fehlzähler zurück.
        Wird aufgerufen wenn eine Person erkannt wird oder die Mappe geschlossen wird.
        """
        if self._no_person_warning_timer.isActive():
            self._no_person_warning_timer.stop()
        self._missed_presence_checks = 0
        self._set_reset_button_warning(False)

    def _get_auto_close_missed_check_limit(self):
        """
        Berechnet wie viele verpasste Anwesenheitsprüfungen für Auto-Close nötig sind.

        Beispiel: close_on_no_person_seconds=10, no_person_check_interval_ms=2000
        → 10000ms / 2000ms = 5 verpasste Checks bis Auto-Close

        :return: Anzahl verpasster Checks bis zum automatischen Schließen.
        """
        interval_ms = max(100, int(self.no_person_check_interval_ms))
        timeout_ms = max(5000, int(self.close_on_no_person_seconds) * 1000)
        return max(1, int(round(timeout_ms / float(interval_ms))))

    def _get_warning_start_missed_checks(self):
        """
        Berechnet ab wann der Warn-Blinker aktiviert wird.
        Warnung startet 5 Sekunden vor dem eigentlichen Auto-Close.

        :return: Anzahl verpasster Checks ab der die Warnung beginnt.
        """
        interval_ms = max(100, int(self.no_person_check_interval_ms))
        # 5 Sekunden vor Auto-Close beginnt die Warnung
        warning_checks = max(1, int(round(5000 / float(interval_ms))))
        return max(0, self._get_auto_close_missed_check_limit() - warning_checks)

    def _update_no_person_warning_state(self):
        """
        Aktualisiert den Warnzustand basierend auf der Anzahl verpasster Checks.
        Wenn Schwellenwert erreicht: Reset-Button blinkt. Darunter: Normal.
        """
        if not self._is_open or self.loading_active or self.is_animating:
            # Warnung nur sinnvoll wenn Mappe offen und keine Animation läuft
            self._stop_no_person_timer()
            return

        if self._missed_presence_checks >= self._get_warning_start_missed_checks():
            if not self._no_person_warning_timer.isActive():
                self._set_reset_button_warning(True)
                # Alle 400ms blinken
                self._no_person_warning_timer.start(400)
        else:
            if self._no_person_warning_timer.isActive():
                self._no_person_warning_timer.stop()
            self._set_reset_button_warning(False)

    def _blink_no_person_warning(self):
        """
        Timer-Callback: lässt den Reset-Button pulsieren.
        Wird alle 400ms aufgerufen während der Warn-Phase aktiv ist.
        """
        if not self._is_open or self.loading_active or self.is_animating:
            self._stop_no_person_timer()
            return
        self._set_reset_button_warning(True)

    @pyqtSlot(bool)
    def on_person_presence_changed(self, is_present):
        """
        Callback: wird vom YOLOWorker aufgerufen wenn sich Anwesenheit ändert.

        Bei Anwesenheit: Timer zurücksetzen.
        Bei Abwesenheit: Fehlzähler erhöhen, ggf. Warnung oder Auto-Close auslösen.

        :param is_present: True = Person erkannt, False = keine Person.
        """
        self._last_person_present = bool(is_present)

        if is_present:
            # Person ist wieder da → alle Warnungen und Zähler zurücksetzen
            self._stop_no_person_timer()
            return

        # Person fehlt: nur reagieren wenn Mappe offen und Auto-Close aktiv
        if (
            not self._is_open
            or self.is_animating
            or self.loading_active
            or not self._auto_close_monitoring_enabled
            or not self.close_on_no_person_enabled
        ):
            return

        self._missed_presence_checks += 1
        print(
            f"Auto-close check {self._missed_presence_checks}/"
            f"{self._get_auto_close_missed_check_limit()} missed"
        )
        self._update_no_person_warning_state()

        # Schwellenwert erreicht → Mappe automatisch schließen
        if self._missed_presence_checks >= self._get_auto_close_missed_check_limit():
            self._stop_no_person_timer()
            self.close_folder(reason="auto_close")
