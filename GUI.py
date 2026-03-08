from gui.main_gui import ScalingAkteGUI
from PyQt6.QtWidgets import QApplication
import sys

__all__ = ["ScalingAkteGUI"]


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScalingAkteGUI()
    window.apply_window_state()
    sys.exit(app.exec())
