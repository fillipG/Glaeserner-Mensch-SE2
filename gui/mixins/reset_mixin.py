"""
gui/mixins/reset_mixin.py
--------------------------
Mixin für Reset-Countdown und Button-Zustand der ScalingAkteGUI.

Zuständigkeiten:
- reset_logic: Entscheidet ob Countdown oder sofortiger Reset
- Countdown-Anzeige: Zahl über dem Reset-Button (ResetCountdownItem)
- _set_reset_button_empty: Tauscht Button-Bild gegen leere Variante

Ablauf beim Reset-Klick:
    1. reset_logic() aufgerufen
    2. Wenn Mappe offen: _start_reset_countdown() → zeigt Countdown-Overlay
    3. _update_reset_countdown() zählt jede Sekunde runter
    4. Bei 0: close_folder("manual_countdown") → Mappe schließt

Benötigte self-Attribute (in ScalingAkteGUI.__init__ gesetzt):
    _reset_countdown_timer, _reset_countdown_remaining, _reset_countdown_item
    _reset_button_original_pixmap, _reset_button_empty_path
    reset_countdown_seconds, scene, btn_reset (wenn Mappe offen)
"""

import os

from PyQt6.QtGui import QPixmap

from ..ui_widgets import ResetCountdownItem


class ResetMixin:
    """Mixin: Reset-Countdown und Button-Zustand."""

    def reset_logic(self):
        """
        Haupt-Einstiegspunkt für den Reset-Button.
        Wenn die Mappe offen ist: Countdown starten.
        Wenn die Mappe zu ist (Entwicklermodus): sofort schließen.
        """
        if self._start_reset_countdown():
            return
        self.close_folder(reason="manual")

    def _start_reset_countdown(self):
        """
        Startet den visuellen Reset-Countdown über dem Reset-Button.

        :return: True wenn Countdown aktiv ist oder gestartet wurde,
                 False wenn kein Countdown möglich (Mappe zu).
        """
        if not self._is_open or not hasattr(self, "btn_reset"):
            return False

        # Wenn Countdown bereits läuft: nichts tun
        if self._reset_countdown_timer.isActive():
            return True

        # Button-Bild gegen leere Variante tauschen (visuelles Feedback)
        self._set_reset_button_empty(True)
        seconds = max(1, int(self.reset_countdown_seconds))
        self._reset_countdown_remaining = seconds

        # Altes Countdown-Item entfernen falls vorhanden
        if self._reset_countdown_item is not None:
            self.scene.removeItem(self._reset_countdown_item)
            self._reset_countdown_item = None

        # Countdown-Overlay über dem Reset-Button positionieren
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
        # Jede Sekunde _update_reset_countdown aufrufen
        self._reset_countdown_timer.start(1000)
        return True

    def _update_reset_countdown(self):
        """
        Timer-Callback: Zählt den Countdown um 1 herunter.
        Bei 0: Countdown-Anzeige entfernen und Mappe schließen.
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
        Entfernt das Countdown-Overlay und stellt den Reset-Button wieder her.
        Wird aufgerufen wenn der Countdown fertig ist oder abgebrochen wird.
        """
        if self._reset_countdown_item is not None:
            self.scene.removeItem(self._reset_countdown_item)
            self._reset_countdown_item = None
        self._set_reset_button_empty(False)

    def _set_reset_button_empty(self, is_empty):
        """
        Tauscht das Reset-Button-Bild gegen die leere Variante (und zurück).
        Die leere Variante signalisiert dem Nutzer, dass gerade ein Countdown läuft.

        :param is_empty: True = leere Variante zeigen, False = Normal-Zustand.
        """
        button = getattr(self, "btn_reset", None)
        if button is None:
            return
        try:
            if is_empty:
                # Original-Pixmap sichern bevor ausgetauscht wird
                if self._reset_button_original_pixmap is None:
                    self._reset_button_original_pixmap = button.current_pixmap
                if os.path.exists(self._reset_button_empty_path):
                    empty = QPixmap(self._reset_button_empty_path)
                    button.pixmap1 = empty
                    button.pixmap2 = empty
                    button.current_pixmap = empty
                    button.update()
            else:
                # Original-Pixmap wiederherstellen
                if self._reset_button_original_pixmap is not None:
                    button.pixmap1 = self._reset_button_original_pixmap
                    button.pixmap2 = self._reset_button_original_pixmap
                    button.current_pixmap = self._reset_button_original_pixmap
                    button.update()
                self._reset_button_original_pixmap = None
        except RuntimeError:
            # Qt-Objekt wurde bereits zerstört (z.B. beim App-Beenden)
            self._reset_button_original_pixmap = None
