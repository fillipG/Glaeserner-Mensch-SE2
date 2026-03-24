"""
gui/mixins/camera_mixin.py
---------------------------
Mixin für Kamera-Preview und Gesichtserkennung im Vorschaubild.

Zuständigkeiten:
- on_camera_frame: Live-Kamerabild empfangen, Gesichter erkennen und anzeigen
- _filter_preview_faces: Haar-Cascade-Ergebnisse bereinigen (Duplikate, Seitengesichter)
- _freeze_camera_preview: Kamerabild einfrieren wenn Foto aufgenommen wurde

Technischer Hinweis:
    Die Gesichtserkennung im Preview nutzt absichtlich Haar-Cascades (nicht YOLO),
    weil sie im GUI-Thread schneller ist und nur für die visuelle Bounding-Box-Anzeige
    gebraucht wird. YOLO läuft separat im YOLOWorker-Thread für die eigentliche Erkennung.

Benötigte self-Attribute (in ScalingAkteGUI.__init__ gesetzt):
    camera_pixmap_item, _face_cascade, _face_detect_interval_ms
    _last_face_detect_ms, _last_faces, _face_detection_scale
    _cam_display_w, _cam_display_h, _last_camera_preview_pixmap
    _live_deepface_svc, _is_open, is_animating, loading_active
    current_language

Autor: Lukas Käuper (Kamera-Integration), Florian Hoeft (Bug-Fixes)
"""

import time
import cv2

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage, QColor


class CameraMixin:
    """Mixin: Kamera-Preview und Gesichtserkennung."""

    def _freeze_camera_preview(self):
        """
        Friert das Live-Kamerabild ein.
        Wird aufgerufen wenn ein Foto gemacht wurde, damit der Nutzer sieht,
        welches Bild gerade analysiert wird.
        Falls kein letztes Bild vorhanden ist, wird ein schwarzer Platzhalter gezeigt.
        """
        if self.camera_pixmap_item is None:
            return
        if self._last_camera_preview_pixmap is not None:
            self.camera_pixmap_item.setPixmap(self._last_camera_preview_pixmap)
            return
        # Kein Bild vorhanden → schwarzer Fallback
        placeholder = QPixmap(self._cam_display_w, self._cam_display_h)
        placeholder.fill(QColor("black"))
        self.camera_pixmap_item.setPixmap(placeholder)

    def _filter_preview_faces(self, faces, frame_shape):
        """
        Bereinigt die Haar-Cascade-Treffer für das Live-Overlay.

        Filtert heraus:
        - Zu kleine Erkennungen (< 8% des Frames)
        - Schiefe Gesichter (Seitenverhältnis außerhalb 0.7-1.35)
        - Gesichter am Bildrand (können Artefakte sein)
        - Duplikate (mehrere Boxen für dasselbe Gesicht)

        :param faces: Liste von (x, y, w, h) Rechtecken von detectMultiScale.
        :param frame_shape: (height, width, channels) des Originalframes.
        :return: Bereinigte Liste von (x, y, w, h) Rechtecken.
        """
        frame_h, frame_w = frame_shape[:2]
        min_face_size = max(60, int(min(frame_h, frame_w) * 0.08))
        border_margin = max(8, int(min(frame_h, frame_w) * 0.01))

        filtered_faces = []
        for (x, y, w, h) in faces:
            x, y, w, h = int(x), int(y), int(w), int(h)
            # Zu kleine Treffer ignorieren
            if w < min_face_size or h < min_face_size:
                continue
            # Seitenverhältnis prüfen (echte Gesichter sind annähernd quadratisch)
            aspect_ratio = w / float(max(h, 1))
            if aspect_ratio < 0.7 or aspect_ratio > 1.35:
                continue
            # Randtreffer ignorieren
            if x <= border_margin or y <= border_margin:
                continue
            if (x + w) >= (frame_w - border_margin) or (y + h) >= (frame_h - border_margin):
                continue
            filtered_faces.append((x, y, w, h))

        # Nach Größe sortieren, größte zuerst
        filtered_faces.sort(key=lambda box: box[2] * box[3], reverse=True)

        # Duplikate entfernen: kleinere Box ignorieren wenn Mittelpunkt in größerer liegt
        deduplicated_faces = []
        for candidate in filtered_faces:
            cx = candidate[0] + candidate[2] / 2.0
            cy = candidate[1] + candidate[3] / 2.0
            is_duplicate = False
            for existing in deduplicated_faces:
                ex, ey, ew, eh = existing
                if ex <= cx <= (ex + ew) and ey <= cy <= (ey + eh):
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduplicated_faces.append(candidate)

        return deduplicated_faces

    @pyqtSlot(object)
    def on_camera_frame(self, frame):
        """
        Callback: wird für jedes neue Kamera-Frame aus dem YOLOWorker aufgerufen.

        Führt Gesichtserkennung durch (mit Haar-Cascades, rate-limited auf 50ms),
        zeichnet Bounding-Boxes und optionale Live-DeepFace-Overlays ein,
        und aktualisiert das Kamerabild in der Szene.

        :param frame: BGR-Numpy-Array (OpenCV-Format).
        """
        if self.camera_pixmap_item is None:
            return
        try:
            display_frame = frame

            if self._face_cascade is not None and not self._face_cascade.empty():
                now_ms = int(time.perf_counter() * 1000)
                # Rate-Limiting: Gesichtserkennung läuft nicht bei jedem Frame,
                # sondern maximal alle 50ms (ca. 20x/Sek), um die GUI nicht zu überlasten.
                if (now_ms - self._last_face_detect_ms) >= self._face_detect_interval_ms:
                    self._last_face_detect_ms = now_ms
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Bild verkleinern für schnellere Cascade-Detection
                    scale = max(0.25, min(1.0, float(self._face_detection_scale)))
                    if scale < 1.0:
                        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    else:
                        small = gray
                    faces = self._face_cascade.detectMultiScale(
                        small,
                        scaleFactor=1.15,
                        minNeighbors=8,
                        minSize=(45, 45),
                    )
                    # Koordinaten zurück auf Original-Auflösung skalieren
                    if scale < 1.0 and len(faces) > 0:
                        faces = [
                            (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
                            for (x, y, w, h) in faces
                        ]
                    self._last_faces = self._filter_preview_faces(faces, frame.shape)

                faces = self._last_faces or []

                # Live-DeepFace ist ein reines Preview-Feature und läuft nur im Hintergrund
                self._live_deepface_svc.maybe_trigger(
                    frame, faces,
                    is_open=self._is_open,
                    is_animating=self.is_animating,
                    loading_active=self.loading_active,
                )

                if len(faces) > 0:
                    display_faces = faces[:self._live_deepface_svc.max_faces]
                    display_frame = frame.copy()
                    box_color = (188, 228, 244)  # Helles Blau für Bounding-Boxes
                    for (x, y, w, h) in display_faces:
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_color, 2)

                    # Live-DeepFace-Overlay-Texte (Age, Gender, Emotion) einzeichnen
                    overlay_data = self._live_deepface_svc.get_overlay_data(
                        display_faces, language=self.current_language,
                    )
                    for overlay in overlay_data:
                        text = overlay["text"]
                        text_x = overlay["x"]
                        text_y = max(overlay["y"] - 12, 20)
                        font_scale = 0.9
                        thickness = 2
                        (text_w, text_h), baseline = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness,
                        )
                        # Dunkler Hintergrund für bessere Lesbarkeit
                        cv2.rectangle(
                            display_frame,
                            (text_x - 6, text_y - text_h - 6),
                            (text_x + text_w + 6, text_y + baseline + 4),
                            (20, 20, 20), -1,
                        )
                        cv2.putText(
                            display_frame, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (80, 220, 255), thickness, cv2.LINE_AA,
                        )

            # BGR → RGB konvertieren und als Qt-Pixmap in die Szene schreiben
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            q_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(q_img).scaled(
                self._cam_display_w, self._cam_display_h,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            self._last_camera_preview_pixmap = pixmap
            self.camera_pixmap_item.setPixmap(pixmap)

        except Exception as e:
            print(f"on_camera_frame Fehler: {e}")
