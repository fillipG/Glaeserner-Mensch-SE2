import cv2
import numpy as np
import os


def create_advanced_sketch(image_path, output_path):
    # 1. Bild laden
    img = cv2.imread(image_path)
    if img is None:
        print("Fehler: Bild konnte nicht geladen werden. Prüfe den Pfad!")
        return

    # 2. Vorbereitung: In Graustufen umwandeln
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. LOKALER KONTRAST (CLAHE) - Das betont Nase, Augen und Mund extrem gut
    # clipLimit: Höherer Wert = mehr Kontrast (standard 2.0 bis 4.0)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)

    # 4. GAMMA-KORREKTUR - Dunkelt Mitteltöne ab für mehr Tiefe
    # gamma < 1.0 macht das Bild kontrastreicher in den Schatten
    gamma = 0.8
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gray_final = cv2.LUT(gray_enhanced, table)

    # 5. SKIZZEN-PROZESS (Color Dodge)
    # Invertieren
    inverted_img = 255 - gray_final

    # Weichzeichnen (Blur)
    # Ein Wert von (31, 31) gibt oft schönere Schattierungen als (21, 21)
    blurred_img = cv2.GaussianBlur(inverted_img, (31, 31), 0)

    # Erneut invertieren
    inverted_blurred = 255 - blurred_img

    # Division für den Sketch-Effekt
    sketch = cv2.divide(gray_final, inverted_blurred, scale=256.0)

    # 6. ERGEBNIS SPEICHERN
    cv2.imwrite(output_path, sketch)
    print(f"Optimierte Skizze gespeichert unter: {output_path}")


# --- ANWENDUNG ---
# Pfade anpassen (Nutze r"PFAD" für Windows-Pfade mit Backslashes)
input_file = r"D:/Downloads/face4.jpg"
output_file = r"D:/Downloads/face4_sketch.jpg"

create_advanced_sketch(input_file, output_file)