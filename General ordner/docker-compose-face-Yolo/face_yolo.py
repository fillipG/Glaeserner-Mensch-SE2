import cv2
from ultralytics import YOLO
from rembg import remove, new_session
from PIL import Image
import os
import time
import yaml

# Einstellungen
INPUT_DIR = "main_image"
YAML_PATH = "final/faces_log.yaml"  # Pfad zur YAML-Datei
SKETCH_DIR = "sketch" # Pfad zum Ordner für die Sketch Bilder
DEEPFACE_INBOX = "deepface_inbox" # Inbox für Deepface
MOONDREAM_INBOX = "moondream_inbox" # Inbox für Moondream

confidence = 0.7  # Ab welcher Konfidenz ein Gesicht erkannt wird
padding = 40     # Zusätzlicher Rand, verbessert das entfernen des Hintergrunds.

# 1. YOLO-Modell laden (einmalig, außerhalb der Schleife)
model = YOLO("yolov8n-face.pt")
model.to('cuda')

# 2. Session für GPU erstellen
# Falls keine GPU gefunden wird, nutzt es automatisch die CPU.
rembg_session = new_session("u2net")

print("Warte auf Bilder im Ordner 'main_image'... ")

# Dauerhaft den Ordner überwachen
while True:
    # Bild im Ordner suchen
    image_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ]

    if image_files:
        image_name = image_files[0]
        image_path = os.path.join(INPUT_DIR, image_name)
        print(f"Bild gefunden: {image_path} – wird verarbeitet...")

        # Bild laden
        image = cv2.imread(image_path)

        # Gesichter erkennen
        results = model(image, conf=confidence, device='cuda')

        # Anzahl erkannter Gesichter über YOLO
        face_count = len(results[0].boxes)
        print(f"  Erkannte Gesichter: {face_count}")

        # In YAML schreiben
        with open(YAML_PATH, "w") as f:
            yaml.dump({"face_count": face_count}, f)

        # Sicherstellen, dass alle Zielordner existieren
        for folder in [SKETCH_DIR, DEEPFACE_INBOX, MOONDREAM_INBOX]:
            os.makedirs(folder, exist_ok=True)

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
                face_no_bg = remove(face_pil, session=rembg_session)

                # --- VERTEILUNG AN DIE INBOXEN ---
                file_name = f"face{face_nr}.png"

                # 1. Speichern für die GUI
                face_no_bg.save(os.path.join(SKETCH_DIR, file_name))

                # 2. Speichern für Deepface (Deepface löscht dieses File nach der Arbeit)
                face_no_bg.save(os.path.join(DEEPFACE_INBOX, file_name))

                # 3. Speichern für Moondream (Moondream löscht dieses File nach der Arbeit)
                face_no_bg.save(os.path.join(MOONDREAM_INBOX, file_name))

                print(f"  Gesicht {face_nr} an alle Inboxes verteilt.")
                face_nr += 1

        # 3. Das Originalbild löschen, wenn alle Gesichter verteilt sind
        try:
            os.remove(image_path)
            print(f"Fertig! Originalbild {image_name} gelöscht. Warte auf Analyse...")
        except Exception as e:
            print(f"Fehler beim Löschen des Originalbilds: {e}")

    # 0.5 Sekunde warten, dann erneut prüfen
    time.sleep(0.5)