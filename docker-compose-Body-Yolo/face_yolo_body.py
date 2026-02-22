# pip install ultralytics opencv-python rembg pillow
import cv2
from ultralytics import YOLO
from rembg import remove
from PIL import Image
import os

# Eingabebild
INPUT_DIR = "main_image"
image_path = next(
    os.path.join(INPUT_DIR, f)       # Baut den Pfad zum Bild 
    for f in os.listdir(INPUT_DIR)   # Sucht die Bilddatei im Ordner main_image
    if f.endswith(('.png', '.jpg', '.jpeg'))
)

# Einstellungen
confidence = 0.5  # Ab welcher Konfidenz ein Gesicht erkannt wird 
padding = 0  # Zusätzlicher Rand falls nötig, verbessert das entfernen des Hintergrunds.

# Ausgabe-Ordner erstellen
if not os.path.exists("faces_yolo_body"):
    os.makedirs("faces_yolo_body")

# YOLO-Modell laden
model = YOLO("yolov8n.pt")

# Bild laden
image = cv2.imread(image_path)

# Gesichter erkennen
results = model(image, conf=confidence)

# Für jedes erkannte Gesicht
face_nr = 1
for result in results:
    for box in result.boxes:
        # Koordinaten vom Gesicht
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Rand hinzufügen, aber innerhalb der Bildgrenzen 
        height, width = image.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)

        # Gesicht ausschneiden
        face = image[y1:y2, x1:x2]
        
        # Zu RGB konvertieren
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        
        # Hintergrund entfernen
        face_no_bg = remove(face_pil)
        
        # Speichern
        face_no_bg.save(f"faces_yolo_body/face{face_nr}.png")
        
        face_nr += 1

print(f"Fertig! Gesichter mit Körper ausgeschnitten")