import os
import time
from datetime import datetime
import yaml
import cv2

def run_yolo():
    os.makedirs("main_image", exist_ok=True)    # Erstellen des Ordners 

    timestamp = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")

    camera = cv2.VideoCapture(0) # Öffnen der Kamera 
    ret, frame = camera.read()  # Aufnehmen 

    # boolean ret: True, wenn das Bild erfolgreich aufgenommen wurde
    if ret:
        cv2.imwrite(f"main_image/main_{timestamp}.jpg", frame)
        print(f"Bild gespeichert: main_image/main_{timestamp}.jpg")
    else:
        print("Bild konnte nicht gelesen werden")
    

def stream_video():
    camera = cv2.VideoCapture(0) # Öffnen der Kamera 

    while True:
        ret, frame = camera.read()  # Aufnehmen 
        if not ret:
            print("Fehler beim Lesen des Videoframes")
            break

        cv2.imshow("Live Stream", frame)  # Anzeigen des Videoframes

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Beenden mit 'q'
            break

    camera.release()
    cv2.destroyAllWindows()    
    

class PipelineAggregator:
    def __init__(self, config_data):
        # 1. Identify which models we are waiting for from config.yaml
        self.enabled_models = [cfg for cfg in config_data["pipeline"] if cfg.get("enabled", False)]
        self.required_ids = [m["id"] for m in self.enabled_models]

        # 2. Folder settings
        self.watch_dir = os.path.abspath(self.enabled_models[0]["watch_dir"])
        self.file_ext = ".yaml"

        # 3. Wait Room for grouping results
        self.results_cache = {}

        # 4. Initial sweep to ignore old files
        self.seen_files = {f for f in os.listdir(self.watch_dir) if f.endswith(self.file_ext)}
        os.makedirs(self.watch_dir, exist_ok=True)

        print(f"🚀 AGGREGATOR STARTING")
        print(f"📡 Waiting for models: {', '.join(self.required_ids).upper()}")
        print(f"📂 Watching folder: {self.watch_dir}\n")

    def check_for_updates(self):
        try:
            current_files = {f for f in os.listdir(self.watch_dir) if f.endswith(self.file_ext)}
            new_files = current_files - self.seen_files

            for file_name in new_files:
                time.sleep(0.1)  # Buffer for file writing
                self.process_incoming_file(file_name)
                self.seen_files.add(file_name)
        except Exception as e:
            print(f"⚠️ Error: {e}")

    def process_incoming_file(self, file_name):
        try:
            # Expected format: face1_moondream.yaml
            name_no_ext = file_name.replace(self.file_ext, "")
            if "_" not in name_no_ext: return

            base_id, model_id = name_no_ext.split("_", 1)

            if base_id not in self.results_cache:
                self.results_cache[base_id] = {}

            # Read result
            file_path = os.path.join(self.watch_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                # Store the content
                self.results_cache[base_id][model_id] = data.get("description") if isinstance(data, dict) else data

            # --- PROGRESS UPDATE ---
            received = list(self.results_cache[base_id].keys())
            waiting_for = [m for m in self.required_ids if m not in received]

            print(f"📥 [{base_id.upper()}] Received: {model_id.upper()}")

            if waiting_for:
                print(f"   ⏳ Still waiting for: {', '.join(waiting_for).upper()}")
            else:
                self.finalize_group(base_id)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    def finalize_group(self, base_id):
        print(f"\n✅ [COMPLETE ANALYSIS] {base_id.upper()}")
        print("=" * 60)

        for m_id in self.required_ids:
            content = self.results_cache[base_id][m_id]
            print(f"🤖 {m_id.upper()}: {content}")

        print("=" * 60 + "\n")
        # Clear cache for this ID
        del self.results_cache[base_id]


def run_pipeline():
    try:
        with open("config.yaml", "r") as f:
            config_data = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Config Error: {e}")
        return
    
    aggregator = PipelineAggregator(config_data)

    try:
        while True:
            aggregator.check_for_updates()
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    run_yolo()
    #stream_video()
    run_pipeline()