from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import QTimer, pyqtSignal


class TypewriterLabel(QLabel):
    """Label mit Schreibmaschinen-Effekt."""
    finished = pyqtSignal()

    def __init__(self, full_text, interval=30, parent=None):
        super().__init__("", parent)
        self.full_text = full_text
        self.interval = interval
        self.current_index = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._type_char)

    def start_typing(self):
        self.setText("")
        self.current_index = 0
        self._timer.start(self.interval)

    def _type_char(self):
        if self.current_index < len(self.full_text):
            self.current_index += 1
            self.setText(self.full_text[:self.current_index])
        else:
            self._timer.stop()
            self.finished.emit()

