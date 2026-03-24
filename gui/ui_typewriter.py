"""
Name: "ui_typewriter.py"
Beschreibung: Stellt ein Label mit Schreibmaschinen-Effekt bereit.
Autor: Fillip Giffhorn und Dennis Penner
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import QTimer, pyqtSignal

if TYPE_CHECKING:
    from .sound_service import SoundService


class TypewriterLabel(QLabel):
    """Label mit Schreibmaschinen-Effekt."""
    finished = pyqtSignal()

    def __init__(
        self,
        full_text,
        interval=30,
        parent=None,
        sound_service: SoundService | None = None,
    ):
        """
        Initialisiert das Label mit Zieltext und Tippintervall.
        :param full_text: Vollstaendiger Text, der getippt werden soll.
        :param interval: Intervall in Millisekunden pro Zeichen.
        :param parent: Optionales Parent-Widget.
        """
        super().__init__("", parent)
        self.full_text = full_text
        self.interval = interval
        self.current_index = 0
        self._sound_service = sound_service
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._type_char)

    def start_typing(self):
        """
        Startet den Schreibmaschinen-Effekt von vorne.
        """
        self.setText("")
        self.current_index = 0
        self._timer.start(self.interval)

    def _type_char(self):
        """
        Fuegt das naechste Zeichen hinzu oder beendet den Effekt.
        """
        if self.current_index < len(self.full_text):
            char = self.full_text[self.current_index]
            self.current_index += 1
            self.setText(self.full_text[:self.current_index])
            if self._sound_service:
                self._sound_service.play("typewriter", char)
        else:
            self._timer.stop()
            self.finished.emit()
