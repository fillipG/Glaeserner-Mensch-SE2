"""
Name: "gui_state.py"
Beschreibung: Definiert die moeglichen GUI-Zustaende als Enum.
Autor: Fillip Giffhorn
"""

from enum import Enum


class GUIState(Enum):
    """
    Enthält alle Zustände der GUI-Statusmaschine.
    """

    IDLE = "idle"
    ANALYZING = "analyzing"
    RESULTS_READY = "results_ready"
    PRESENCE_MONITORING = "presence_monitoring"
    CLOSED = "closed"
    OPENING = "opening"
    OPEN = "open"
    FLIPPING = "flipping"

