import cv2
from ultralytics import YOLO
from rembg import remove, new_session
from PIL import Image
import os
import time
import yaml
import numpy as np
from fnmatch import fnmatch

# Einstellungen
INPUT_DIR = "main_image"
YAML_PATH = "final/faces_log.yaml"  # Pfad zur YAML-Datei
SKETCH_DIR = "sketch" # Pfad zum Ordner für die Sketch Bilder
DEEPFACE_INBOX = "deepface_inbox" # Inbox für Deepface
MOONDREAM_INBOX = "moondream_inbox" # Inbox für Moondream
CONFIG_PATH = "config.yaml"  # Laufzeit-Konfiguration fuer Face-YOLO
DEBUG_DIR = "debug_face_yolo"  # Debug-Bilder fuer problematische Frames

confidence = 0.5  # Ab welcher Konfidenz ein Gesicht erkannt wird
padding = 40     # Zusätzlicher Rand, verbessert das entfernen des Hintergrunds.


def load_runtime_config():
    # Face-YOLO Parameter vor jedem Lauf neu laden, damit config-Aenderungen live greifen
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"  Konnte Face-YOLO Config nicht lesen, nutze Standardwerte: {e}")
        return confidence

    try:
        face_yolo_cfg = cfg.get("face_yolo", {})
        if isinstance(face_yolo_cfg, dict):
            raw_value = face_yolo_cfg.get("confidence", confidence)
        else:
            raw_value = cfg.get("face_yolo_confidence", confidence)
        return max(0.10, min(0.90, float(raw_value)))
    except (TypeError, ValueError):
        return confidence


def cleanup_previous_batch():
    # Alte Batch-Artefakte entfernen, damit face1/face2 nicht mit dem neuen Lauf kollidieren
    targets = {
        SKETCH_DIR: ["face*.png", "face*.jpg", "face*.jpeg"],
        DEEPFACE_INBOX: ["face*.png", "face*.jpg", "face*.jpeg"],
        MOONDREAM_INBOX: ["face*.png", "face*.jpg", "face*.jpeg"],
        "final": ["face*_*.yaml"],
    }

    for folder, patterns in targets.items():
        os.makedirs(folder, exist_ok=True)
        for name in os.listdir(folder):
            if any(fnmatch(name.lower(), pattern.lower()) for pattern in patterns):
                file_path = os.path.join(folder, name)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"  Konnte altes Artefakt nicht löschen: {file_path} ({e})")


def preprocess_for_detection(image):
    # Bild leicht aufhellen und lokal kontraststaerker machen, damit Gesichter stabiler erkannt werden
    brightened = cv2.convertScaleAbs(image, alpha=1.08, beta=8)
    lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(enhanced, -1, sharpen_kernel)


def write_debug_output(image_name, original_image, enhanced_image, current_confidence, results):
    # Problematische Frames sichern, damit 0-face-Faelle spaeter nachvollziehbar bleiben
    os.makedirs(DEBUG_DIR, exist_ok=True)

    original_debug_path = os.path.join(DEBUG_DIR, "last_input.jpg")
    enhanced_debug_path = os.path.join(DEBUG_DIR, "last_enhanced.jpg")
    cv2.imwrite(original_debug_path, original_image)
    cv2.imwrite(enhanced_debug_path, enhanced_image)

    boxes = results[0].boxes if results and len(results) > 0 else None
    confidences = []
    if boxes is not None and boxes.conf is not None:
        confidences = [float(value) for value in boxes.conf.cpu().tolist()]

    mean_brightness = float(cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY).mean())
    max_conf = max(confidences) if confidences else 0.0
    print(f"  [DEBUG] Kein Gesicht erkannt fuer {image_name}")
    print(f"  [DEBUG] confidence={current_confidence:.2f}, max_box_conf={max_conf:.3f}, brightness={mean_brightness:.1f}")
    print(f"  [DEBUG] Debug-Bilder gespeichert: {original_debug_path}, {enhanced_debug_path}")

# 1. YOLO-Modell laden (einmalig, außerhalb der Schleife)
model = YOLO("yolov8n-face.pt")
model.to('cuda')

# 2. Session für GPU erstellen
# Falls keine GPU gefunden wird, nutzt es automatisch die CPU.
rembg_session = new_session("u2net")
last_used_confidence = None

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

        # Vor dem neuen Lauf alte face-Dateien aus Skizze, Inboxes und final entfernen
        cleanup_previous_batch()

        # Bild laden
        image = cv2.imread(image_path)
        if image is None:
            print(f"Bild konnte nicht geladen werden: {image_path}")
            time.sleep(0.5)
            continue

        current_confidence = load_runtime_config()
        if last_used_confidence != current_confidence:
            print(f"  Face-YOLO confidence aktiv: {current_confidence:.2f}")
            last_used_confidence = current_confidence

        # Bild fuer die Gesichtserkennung leicht verbessern
        detection_image = preprocess_for_detection(image)

        # Gesichter erkennen
        results = model(detection_image, conf=current_confidence, device='cuda')

        # Anzahl erkannter Gesichter über YOLO
        face_count = len(results[0].boxes)
        print(f"  Erkannte Gesichter: {face_count}")
        if face_count == 0:
            write_debug_output(image_name, image, detection_image, current_confidence, results)

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
