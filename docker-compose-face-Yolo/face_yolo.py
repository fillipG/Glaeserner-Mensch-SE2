import cv2
from ultralytics import YOLO
from rembg import remove, new_session
from PIL import Image
import os
import shutil

# --- KONFIGURATION ---
INPUT_DIR = "main_image"
OUTPUT_DIR = "faces_yolo"
CONFIDENCE = 0.5
PADDING_RATIO = 0.4  # Proportionaler Rand basierend auf der Gesichtsgröße

# --- 1. AUSGABEORDNER LEEREN (Docker-kompatibel) ---
# Erstellt den Ordner falls er fehlt oder löscht nur den Inhalt,
# um 'Device or resource busy' Fehlermeldungen in Docker zu vermeiden.
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
else:
    for filename in os.listdir(OUTPUT_DIR):
        file_path = os.path.join(OUTPUT_DIR, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Fehler beim Bereinigen von {file_path}: {e}")

# --- 2. EINGABEBILD SUCHEN ---
files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
if not files:
    print(f"Abbruch: Keine Bilddateien in {INPUT_DIR} gefunden.")
    exit()

# Verarbeitet die erste gefundene Bilddatei
image_path = os.path.join(INPUT_DIR, files[0])
print(f"Analysiere Datei: {image_path}")

# --- 3. MODELLE INITIALISIEREN ---
# Lädt das YOLO-Gesichtsmodell
model = YOLO("yolov8n-face.pt")

# Erstellt eine rembg-Sitzung für bessere Performance
rembg_session = new_session()

# --- 4. BILDVERARBEITUNG ---
image = cv2.imread(image_path)
if image is None:
    print("Fehler: Bilddatei konnte nicht geladen werden.")
    exit()

height, width = image.shape[:2]
results = model(image, conf=CONFIDENCE)

face_nr = 1
for result in results:
    for box in result.boxes:
        # Extraktion der Koordinaten
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        # Berechnung des dynamischen Randes
        face_w = x2 - x1
        face_h = y2 - y1
        pad_w = int(face_w * PADDING_RATIO)
        pad_h = int(face_h * PADDING_RATIO)

        # Zuschnitt berechnen und Bildgrenzen einhalten
        nx1 = max(0, int(x1 - pad_w))
        ny1 = max(0, int(y1 - pad_h))
        nx2 = min(width, int(x2 + pad_w))
        ny2 = min(height, int(y2 + pad_h))

        # Bild ausschneiden und konvertieren
        face_crop = image[ny1:ny2, nx1:nx2]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)

        # Hintergrund mittels rembg entfernen
        print(f"Verarbeite Gesicht Nummer {face_nr}...")
        face_no_bg = remove(face_pil, session=rembg_session)

        # Speichervorgang als PNG
        output_filename = f"face_{face_nr}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        face_no_bg.save(output_path)
        print(f"Datei gespeichert: {output_path}")
        
        face_nr += 1

print(f"Verarbeitung abgeschlossen. {face_nr - 1} Gesichter extrahiert.")