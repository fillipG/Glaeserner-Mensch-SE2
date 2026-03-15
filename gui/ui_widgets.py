"""
Name: "ui_widgets.py"
Beschreibung: Enthält wiederverwendbare QGraphics-Widgets fuer Buttons, Spinner und Countdowns.
Autor: Fillip Giffhorn
"""

from PyQt6.QtWidgets import QGraphicsObject
from PyQt6.QtGui import QPixmap, QColor, QPainter, QPen, QFont
from PyQt6.QtCore import Qt, QRectF, QTimer, pyqtSignal


class AnimatedGraphicsButton(QGraphicsObject):
    """
    Klickbarer Grafikbutton mit optionalem Toggle-Pixmap.
    """

    clicked = pyqtSignal()

    def __init__(self, image1_path, image2_path=None, scale=1.0, parent=None):
        """
        Initialisiert den Grafikbutton mit einem oder zwei Bildern.
        :param image1_path: Pfad zum Standardbild.
        :param image2_path: Optionaler Pfad fuer das Toggle-Bild.
        :param scale: Skalierungsfaktor.
        :param parent: Optionales Parent-Item.
        """
        super().__init__(parent)
        self.pixmap1 = QPixmap(image1_path)
        self.pixmap2 = QPixmap(image2_path) if image2_path else self.pixmap1
        self.current_pixmap = self.pixmap1
        self.scale = float(scale)
        self.is_toggled = False
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)

    def boundingRect(self):
        """
        Liefert den darzustellenden Bereich des Buttons.
        :return: Rechteck fuer die Darstellung.
        """
        if self.current_pixmap.isNull():
            return QRectF(0, 0, 0, 0)
        return QRectF(0, 0, self.current_pixmap.width() * self.scale,
                      self.current_pixmap.height() * self.scale)

    def paint(self, painter, option, widget=None):
        """
        Zeichnet den aktuellen Button-Zustand.
        :param painter: QPainter der Szene.
        :param option: Style-Optionen.
        :param widget: Optionales Ziel-Widget.
        """
        if self.current_pixmap.isNull():
            return
        target = self.boundingRect()
        source = QRectF(self.current_pixmap.rect())
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawPixmap(target, self.current_pixmap, source)

    def mousePressEvent(self, event):
        """
        Verarbeitet Klicks und emittiert das clicked-Signal.
        :param event: Maus-Event.
        """
        if self.pixmap2 and self.pixmap2.cacheKey() != self.pixmap1.cacheKey():
            self.is_toggled = not self.is_toggled
            self.current_pixmap = self.pixmap2 if self.is_toggled else self.pixmap1
            self.update()
        self.clicked.emit()
        super().mousePressEvent(event)


class CircularTimerItem(QGraphicsObject):
    """Runder Countdown-Overlay mit modernem Ring-Design."""

    def __init__(self, duration_s, diameter=220, parent=None):
        """
        Initialisiert den Ring-Countdown.
        :param duration_s: Gesamtdauer in Sekunden.
        :param diameter: Durchmesser in Pixeln.
        :param parent: Optionales Parent-Item.
        """
        super().__init__(parent)
        self.duration_s = max(1, int(duration_s))
        self.diameter = int(diameter)
        self.progress = 0.0  # 0.0 .. 1.0
        self.remaining_s = self.duration_s
        self.setZValue(200)

    def boundingRect(self):
        """
        Liefert den darzustellenden Bereich des Countdowns.
        :return: Rechteck fuer die Darstellung.
        """
        return QRectF(0, 0, self.diameter, self.diameter)

    def set_progress(self, progress, remaining_s):
        """
        Aktualisiert Fortschritt und Restzeit.
        :param progress: Fortschritt zwischen 0.0 und 1.0.
        :param remaining_s: Verbleibende Sekunden.
        """
        self.progress = max(0.0, min(1.0, float(progress)))
        self.remaining_s = max(0, int(round(remaining_s)))
        self.update()

    def paint(self, painter, option, widget=None):
        """
        Zeichnet Ring und Restzeittext.
        :param painter: QPainter der Szene.
        :param option: Style-Optionen.
        :param widget: Optionales Ziel-Widget.
        """
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.boundingRect().adjusted(10, 10, -10, -10)

        bg_pen = QPen(QColor(120, 96, 72, 160), 20)
        bg_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(bg_pen)
        painter.drawEllipse(rect)

        fg_pen = QPen(QColor(244, 228, 188, 230), 20)
        fg_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(fg_pen)
        start_angle = 90 * 16
        span_angle = -int(360 * 16 * self.progress)
        painter.drawArc(rect, start_angle, span_angle)

        painter.setPen(QColor(244, 228, 188))
        font = QFont("Graduate", 35, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(self.boundingRect(), Qt.AlignmentFlag.AlignCenter, f"{self.remaining_s}s")


class LoadingSpinnerItem(QGraphicsObject):
    """Einfacher, typischer Lade-Spinner (animierter Kreisbogen)."""

    def __init__(self, diameter=120, color=QColor(244, 228, 188, 230), direction=1, parent=None):
        """
        Initialisiert den rotierenden Lade-Spinner.
        :param diameter: Durchmesser in Pixeln.
        :param color: Farbe des Spinners.
        :param direction: Drehrichtung, positiv oder negativ.
        :param parent: Optionales Parent-Item.
        """
        super().__init__(parent)
        self.diameter = int(diameter)
        self._angle = 0
        self._color = color
        self._direction = 1 if direction >= 0 else -1
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)
        self.setZValue(200)

    def boundingRect(self):
        """
        Liefert den darzustellenden Bereich des Spinners.
        :return: Rechteck fuer die Darstellung.
        """
        return QRectF(0, 0, self.diameter, self.diameter)

    def _tick(self):
        """
        Dreht den Spinner um einen Schritt weiter.
        """
        self._angle = (self._angle + (8 * self._direction)) % 360
        self.update()

    def stop(self):
        """
        Stoppt die interne Spinner-Animation.
        """
        if self._timer.isActive():
            self._timer.stop()

    def paint(self, painter, option, widget=None):
        """
        Zeichnet den Spinnerbogen.
        :param painter: QPainter der Szene.
        :param option: Style-Optionen.
        :param widget: Optionales Ziel-Widget.
        """
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.boundingRect().adjusted(10, 10, -10, -10)
        pen = QPen(self._color, 14)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        start_angle = int(self._angle * 16)
        span_angle = int(-270 * 16)
        painter.drawArc(rect, start_angle, span_angle)


class ResetCountdownItem(QGraphicsObject):
    """Einfacher Countdown-Text ohne zusaetzliche Animation."""

    def __init__(self, diameter=80, color=QColor(80, 160, 255, 230), parent=None):
        """
        Initialisiert das Reset-Countdown-Overlay.
        :param diameter: Durchmesser in Pixeln.
        :param color: Textfarbe.
        :param parent: Optionales Parent-Item.
        """
        super().__init__(parent)
        self.diameter = int(diameter)
        self.remaining = 0
        self._color = color
        self.setZValue(210)

    def boundingRect(self):
        """
        Liefert den darzustellenden Bereich des Countdowns.
        :return: Rechteck fuer die Darstellung.
        """
        return QRectF(0, 0, self.diameter, self.diameter)

    def set_remaining(self, remaining):
        """
        Setzt die verbleibende Countdown-Zahl.
        :param remaining: Restzeit als Zahl.
        """
        self.remaining = int(max(0, remaining))
        self.update()

    def paint(self, painter, option, widget=None):
        """
        Zeichnet die verbleibende Zahl mittig ins Overlay.
        :param painter: QPainter der Szene.
        :param option: Style-Optionen.
        :param widget: Optionales Ziel-Widget.
        """
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.boundingRect()
        font_size = max(12, int(self.diameter * 0.45))
        painter.setFont(QFont("Graduate", font_size, QFont.Weight.Bold))
        painter.setPen(self._color)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"{self.remaining}")
