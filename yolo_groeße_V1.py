"""
 Anleitung zur Ausführung

1) Benötigte Pakete installieren:
    pip install ultralytics opencv-python numpy

2) Skript starten:
    python yolo_groeße_V1.py

3) Kamera öffnet sich automatisch.
    - ESC drücken, um das Programm zu beenden.
"""

from ultralytics import YOLO
import cv2
import numpy as np
import time

distance = 3.0
factor = 0.158

# ===== Timer Variablen =====
duration = 3
person_start_time = None
image_saved = False
last_capture_time = None
# ===========================

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    yolo_img = results[0].plot()
    r = results[0]

    current_time = time.time()
    frame_height, frame_width = frame.shape[:2]

    person_complete = False

    # ===============================
    # PERSON ERKANNT?
    # ===============================
    if r.keypoints is not None and len(r.keypoints.xy) > 0:

        person = r.keypoints.xy[0]

        # Prüfen ob Kopf + beide Füße existieren
        if len(person) > 16:
            head = person[0]
            left_foot = person[15]
            right_foot = person[16]

            if (head[0] > 0 and head[1] > 0 and
                left_foot[0] > 0 and left_foot[1] > 0 and
                right_foot[0] > 0 and right_foot[1] > 0):

                margin = 20
                if (margin < head[0] < frame_width - margin and
                    margin < head[1] < frame_height - margin and
                    margin < left_foot[1] < frame_height - margin and
                    margin < right_foot[1] < frame_height - margin):

                    person_complete = True

    # ===============================
    # TIMER + COUNTDOWN
    # ===============================
    if person_complete:

        if person_start_time is None:
            person_start_time = current_time

        elapsed = current_time - person_start_time
        remaining = duration - elapsed

        # Countdown anzeigen
        if remaining > 0:
            cv2.putText(yolo_img,
                        f"Foto in: {int(remaining) + 1}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 255), 3)

        # Foto speichern
        elif not image_saved:
            filename = f"person_detected_{int(current_time)}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Bild gespeichert: {filename}")

            image_saved = True
            last_capture_time = current_time

    else:
        person_start_time = None
        image_saved = False

    # ===============================
    # FOTO-AUFGENOMMEN ANZEIGE
    # ===============================
    if last_capture_time is not None:
        if current_time - last_capture_time < 2:
            cv2.putText(yolo_img,
                        "FOTO AUFGENOMMEN!",
                        (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3)

    # ===============================
    # GRÖSSENBERECHNUNG
    # ===============================
    if person_complete:

        head_x, head_y = head
        foot = left_foot if left_foot[1] > right_foot[1] else right_foot
        foot_x, foot_y = foot

        pixel_height = abs(float(foot_y) - float(head_y))
        height_cm = pixel_height * distance * factor

        cv2.putText(yolo_img, f"{height_cm:.1f} cm",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

    else:
        cv2.putText(yolo_img,
                    "Person nicht komplett sichtbar",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

    cv2.imshow("YOLO Pose - Groesse", yolo_img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
