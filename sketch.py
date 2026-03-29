"""
Name: "sketch.py"
Beschreibung: Erzeugt aus Bildern eine kontrastreiche Bleistift-Skizze mit optionaler Speicherung.
Autor: Fillip Giffhorn
"""

import cv2
import numpy as np
import os


def _prepare_bgr_and_alpha(image_path_or_img):
    """
    Normalisiert eine Bildquelle auf ein BGR-Bild plus optionale Alpha-Maske.
    :param image_path_or_img: Dateipfad oder bereits geladenes Numpy-Bild.
    :return: Tuple aus (bgr_bild, alpha_maske) oder (None, None) bei ungueltiger Quelle.
    """
    if isinstance(image_path_or_img, (str, os.PathLike)):
        # Unchanged laden, damit Alpha-Kanal bei PNGs erhalten bleibt
        img = cv2.imread(str(image_path_or_img), cv2.IMREAD_UNCHANGED)
    else:
        img = image_path_or_img

    if img is None:
        return None, None

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None

    if img.ndim != 3:
        raise ValueError(f"Ungueltige Bildform: {img.shape}")

    channels = img.shape[2]

    if channels == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None

    if channels == 3:
        return img, None

    if channels == 4:
        bgr = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3].astype(np.float32) / 255.0

        # Transparenz auf weissen Hintergrund komponieren
        white_bg = np.full_like(bgr, 255, dtype=np.float32)
        bgr_composited = (bgr * alpha[..., None] + white_bg * (1.0 - alpha[..., None])).astype(np.uint8)

        alpha_mask = (alpha * 255).astype(np.uint8)
        return bgr_composited, alpha_mask

    raise ValueError(f"Nicht unterstuetzte Kanalanzahl: {channels}")


def create_advanced_sketch(image_path_or_img, output_path=None, delete_input=False):
    """
    Erstellt aus einem Eingabebild eine Skizze und gibt sie als Numpy-Array zurueck.
    :param image_path_or_img: Dateipfad oder bereits geladenes Numpy-Bild.
    :param output_path: Optionaler Speicherpfad fuer das Ergebnis.
    :param delete_input: Loescht die Eingabedatei nach Verarbeitung, falls Pfad uebergeben wurde.
    :return: Skizzenbild als Numpy-Array oder None bei Fehler.
    """
    try:
        img, alpha_mask = _prepare_bgr_and_alpha(image_path_or_img)
    except ValueError as e:
        print(f"Fehler: {e}")
        return None

    if img is None:
        print("Fehler: Bild konnte nicht geladen werden. Pruefe den Pfad!")
        return None

    # 2. Vorbereitung: In Graustufen umwandeln
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. LOKALER KONTRAST (CLAHE) - Das betont Nase, Augen und Mund extrem gut
    # clipLimit: Hoeherer Wert = mehr Kontrast (standard 2.0 bis 4.0)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)

    # 4. GAMMA-KORREKTUR - Dunkelt Mitteltone ab fuer mehr Tiefe
    # gamma < 1.0 macht das Bild kontrastreicher in den Schatten
    gamma = 0.8
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)], dtype=np.uint8)
    gray_final = cv2.LUT(gray_enhanced, table)

    # 5. SKIZZEN-PROZESS (Color Dodge)
    # Invertieren
    inverted_img = 255 - gray_final

    # Weichzeichnen (Blur)
    # Ein Wert von (31, 31) gibt oft schoenere Schattierungen als (21, 21)
    blurred_img = cv2.GaussianBlur(inverted_img, (31, 31), 0)

    # Erneut invertieren
    inverted_blurred = 255 - blurred_img

    # Division fuer den Sketch-Effekt
    sketch = cv2.divide(gray_final, inverted_blurred, scale=256.0)

    # Transparente Ursprungsbereiche sauber weiss halten
    if alpha_mask is not None:
        a = alpha_mask.astype(np.float32) / 255.0
        sketch = (sketch.astype(np.float32) * a + 255.0 * (1.0 - a)).astype(np.uint8)

    # 6. Optional speichern
    if output_path:
        cv2.imwrite(str(output_path), sketch)
        print(f"Optimierte Skizze gespeichert unter: {output_path}")

    # 7. Optional die Eingabedatei loeschen
    if delete_input and isinstance(image_path_or_img, (str, os.PathLike)) and os.path.exists(str(image_path_or_img)):
        os.remove(str(image_path_or_img))
        print(f"Originalbild geloescht: {image_path_or_img}")

    return sketch


if __name__ == "__main__":
    # --- ANWENDUNG ---
    # Pfade anpassen (Nutze r"PFAD" fuer Windows-Pfade mit Backslashes)
    input_file = r"general_ordner/sketch/face1.png"
    output_file = r"C:/Users/Dennis/Downloads/face4_sketch.jpg"
    create_advanced_sketch(input_file, output_file)
