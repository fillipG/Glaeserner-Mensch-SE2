from enum import Enum


class GUIState(Enum):
    IDLE = "idle"
    ANALYZING = "analyzing"
    RESULTS_READY = "results_ready"
    PRESENCE_MONITORING = "presence_monitoring"
    CLOSED = "closed"
    OPENING = "opening"
    OPEN = "open"
    FLIPPING = "flipping"

