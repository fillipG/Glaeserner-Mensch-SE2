"""
 Anleitung zur Ausführung

1) Benötigte Pakete installieren:
    pip install ultralytics opencv-python numpy

2) Skript starten:
    python yolo_groeße_V1.py

3) Kamera öffnet sich automatisch.
    - ESC drücken, um das Programm zu beenden.

Hinweis:
    distance und factor können angepasst werden, 
    um die Genauigkeit der Größenschätzung zu verbessern.

"""


from ultralytics import YOLO
import cv2
import numpy as np

# Kalibrierung des Faktors:
# alter Faktor = 0.4
# echt = 180 cm
# angezeigt = 165 cm
# neuer Faktor = 0.4 * (180/165) = 0.436

distance = 3.0          # Entfernung der Person in Metern
factor = 0.158       # Startwert für den Faktor (angepasst)

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    yolo_img = results[0].plot()

    r = results[0]

    # Falls Keypoints erkannt wurden, Größe berechnen
    if r.keypoints is not None and len(r.keypoints.xy) > 0:
        
        person = r.keypoints.xy[0]

        head_x, head_y = person[0]    # Kopf 
        foot = None

        # Fuß
        for idx in [16, 14, 15, 13]:
            if idx < len(person):
                foot = person[idx]
                break

        if foot is not None:
            foot_x, foot_y = foot

            # Pixelhöhe
            pixel_height = abs(float(foot_y) - float(head_y))

            height_cm = pixel_height * distance * factor
            current_height = height_cm    #speichern der aktuellen Größe

            cv2.putText(yolo_img, f"{height_cm:.1f} cm",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        else:
            cv2.putText(yolo_img, "Fuss nicht erkannt", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    else:
        cv2.putText(yolo_img, "Keine Person erkannt", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("YOLO Pose - Groesse", yolo_img)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
