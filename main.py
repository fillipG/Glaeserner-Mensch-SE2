# --- ANLEITUNG ---
# 1. Starte Docker Desktop
# 2. Führe "docker-compose up -d --build" im Terminal aus
# 3. führe main.py aus
# 4. Kopiere Beispielbilder in "faces_yolo"

import os
import time

# --- NEW CONFIGURATION ---
# These point to folders inside your project directory
INPUT_DIR = os.path.abspath("./faces_yolo")
FINAL_DIR = os.path.abspath("./final")


def run_test_manager():
    print("=" * 50)
    print("MUSEUM SYSTEM ACTIVE")
    print(f"1. Drop images into: {INPUT_DIR}")
    print(f"2. Descriptions will be saved in: {FINAL_DIR}")
    print("=" * 50)

    # Ensure folders exist
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(FINAL_DIR, exist_ok=True)

    # To avoid printing the same file a million times, we keep track of what we've seen
    seen_results = set()

    try:
        while True:
            # Check for .txt files in the 'final' folder
            current_files = [f for f in os.listdir(FINAL_DIR) if f.endswith(".txt")]

            for file in current_files:
                if file not in seen_results:
                    txt_path = os.path.join(FINAL_DIR, file)

                    # Wait a moment to ensure file is written
                    time.sleep(0.2)

                    try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                            description = f.read().strip()

                        print(f"\n[NEW ANALYSIS] File: {file}")
                        print(f"Description: {description}")
                        print("-" * 30)

                        seen_results.add(file)
                    except Exception as e:
                        print(f"Error reading {file}: {e}")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping Manager...")


if __name__ == "__main__":
    run_test_manager()