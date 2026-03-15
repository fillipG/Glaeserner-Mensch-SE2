import os
import yaml
import time
import warnings
import numpy as np
from PIL import Image
from fer import FER
import config

# TensorFlow Logs reduzieren
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)

def scan_image(detector, image_path, output_folder, last_modified):
    filename = os.path.basename(image_path)
    yaml_filename = f"{os.path.splitext(filename)[0]}.yaml"
    yaml_path = os.path.join(output_folder, yaml_filename)

    try:
        mtime = os.path.getmtime(image_path)

        # Nur scannen, wenn neu oder Zeitstempel geändert
        if filename in last_modified and last_modified[filename] == mtime:
            return

        # RGB-Fix für PNG/RGBA
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)

        result = detector.detect_emotions(img_array)

        if result:
            emotions = result[0]["emotions"]
            dominant_emotion = max(emotions, key=emotions.get)
            confidence = float(emotions[dominant_emotion])

            output_data = {
                "Emotion": str(dominant_emotion),
                "Confidence": round(confidence, 4),
                "LastUpdate": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(yaml_path, "w") as f:
                yaml.dump(output_data, f, default_flow_style=False)

            print(f"✅ DATEI VERARBEITET: {yaml_filename} | Emotion: {dominant_emotion}")
        else:
            print(f"⚠️  Kein Gesicht gefunden in: {filename}")

        last_modified[filename] = mtime

    except Exception as e:
        print(f"❌ FEHLER bei {filename}: {e}")

def main():
    try:
        print("--- Emotion Scanner Start ---")
        detector = FER(mtcnn=True)
        print("--- KI-Modell bereit ---")
    except Exception as e:
        print(f"Fehler: {e}")
        return

    last_modified = {}
    source = config.SCAN_FOLDER
    output = config.FINAL_FOLDER

    print(f"🔍 Permanente Überwachung gestartet: {source}")
    print("💡 Tipp: Kopiere jetzt einfach ein Bild in den Ordner...")

    # Diese Schleife läuft EWIG
    while True:
        if os.path.exists(source):
            files = os.listdir(source)
            images = [
                os.path.join(source, f)
                for f in files
                if f.lower().endswith(config.SUPPORTED_FORMATS)
            ]

            if not images:
                # Optional: Ein Punkt alle 10 Sek anzeigen, um zu sehen, dass er noch lebt
                pass

            for img_path in images:
                scan_image(detector, img_path, output, last_modified)
            pass

        # Wartezeit zwischen den Scans (z.B. 0.5 Sek aus config)
        time.sleep(config.SCAN_INTERVAL)

if __name__ == "__main__":
    main()