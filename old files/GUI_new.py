# Zur verwendung bitte "pip install opencv-python pillow numpy" oder unter Windows "py -m pip install opencv-python pillow numpy" ausführen

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import random
import time
import os

# --- KONFIGURATION ---
BACKGROUND_IMAGE_PATH = "../pictures/folder_V1_bearbeitet-1.jpg"

REPORT_WIDTH = 689
REPORT_HEIGHT = 843

REPORT_X_POS = 700
REPORT_Y_POS = 20

# Koordinaten für die Texte (HIER HABE ICH DIE NEUEN FELDER ERGÄNZT)
# Nutze das gelbe Maus-Tool, um die perfekten zahlen zu finden!
TEXT_POSITIONS = {
    "age": (160, 380),
    "gender": (180, 396),
    "emotion": (290, 397),
    "clothing": (200, 412),

    # --- NEUE FELDER (Positionen sind geschätzt, bitte anpassen!) ---
    "height": (310,380),  # Größe
    "threat": (230, 430),  # Gefahrlevel
    "crime": (180, 444)  # Straftat (wahrscheinlich weiter unten auf dem Blatt)
}


class SecurityProfilingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Surveillance System")
        self.root.geometry("1450x1000")

        # --- GUI AUFBAU ---
        self.main_frame = tk.Frame(root, bg="#2c3e50")
        self.main_frame.pack(fill="both", expand=True)

        # 1. Kamera Bereich
        self.cam_frame = tk.LabelFrame(self.main_frame, text="Live Feed & Analysis", bg="black", fg="white")
        self.cam_frame.place(x=20, y=20, width=640, height=500)

        self.video_label = tk.Label(self.cam_frame, bg="black")
        self.video_label.pack(fill="both", expand=True)

        # 2. Report Bereich
        self.report_canvas = tk.Canvas(self.main_frame, bg="gray", highlightthickness=0)
        self.report_canvas.place(x=REPORT_X_POS, y=REPORT_Y_POS, width=REPORT_WIDTH, height=REPORT_HEIGHT)

        self.load_background()

        # Text-Platzhalter erstellen
        self.text_ids = {}
        font_style = ("Courier New", 12, "bold")
        crime_font = ("Courier New", 14, "bold")  # Etwas größer für die Straftat

        # Alte Felder
        self.text_ids["age"] = self.report_canvas.create_text(TEXT_POSITIONS["age"], text="SCANNING...", fill="#333",
                                                              anchor="w", font=font_style)
        self.text_ids["gender"] = self.report_canvas.create_text(TEXT_POSITIONS["gender"], text="WAITING", fill="#333",
                                                                 anchor="w", font=font_style)
        self.text_ids["emotion"] = self.report_canvas.create_text(TEXT_POSITIONS["emotion"], text="---", fill="#333",
                                                                  anchor="w", font=font_style)
        self.text_ids["clothing"] = self.report_canvas.create_text(TEXT_POSITIONS["clothing"], text="ANALYZING",
                                                                   fill="#333", anchor="w", font=font_style)

        # --- NEUE FELDER INITIALISIEREN ---
        self.text_ids["height"] = self.report_canvas.create_text(TEXT_POSITIONS["height"], text="--- cm", fill="#333",
                                                                 anchor="w", font=font_style)

        # Gefahrlevel in ROT für mehr Effekt
        self.text_ids["threat"] = self.report_canvas.create_text(TEXT_POSITIONS["threat"], text="CALCULATING",
                                                                 fill="darkred", anchor="w", font=font_style)

        # Straftat
        self.text_ids["crime"] = self.report_canvas.create_text(TEXT_POSITIONS["crime"], text="", fill="#333",
                                                                anchor="w", font=crime_font)

        # --- ENTWICKLER TOOL ---
        self.mouse_label = tk.Label(self.main_frame, text="Maus: 0 / 0", bg="yellow", font=("Arial", 10, "bold"))
        self.mouse_label.place(x=REPORT_X_POS, y=REPORT_HEIGHT + 30)
        self.report_canvas.bind('<Motion>', self.show_mouse_coords)

        # --- KAMERA SETUP ---
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.last_analysis_time = 0
        self.update_loop()

    def load_background(self):
        try:
            pil_image = Image.open(BACKGROUND_IMAGE_PATH)
            pil_image = pil_image.resize((REPORT_WIDTH, REPORT_HEIGHT), Image.Resampling.LANCZOS)
            self.bg_photo = ImageTk.PhotoImage(pil_image)
            self.report_canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")
        except Exception as e:
            print(f"Fehler: {e}")
            self.report_canvas.create_text(REPORT_WIDTH // 2, REPORT_HEIGHT // 2, text="BILD FEHLT", fill="white")

    def mock_ai_analysis(self):
        """Hier werden die lustigen Daten generiert"""
        genders = ["MALE", "FEMALE", "ALIEN", "CYBORG"]
        emotions = ["ANGRY", "HAPPY", "SUSPICIOUS", "HUNGRY", "CONFUSED"]
        clothes = ["HOODIE", "T-SHIRT", "LAB COAT", "PYJAMAS"]

        # --- NEUE LISTEN ---
        crimes = [
            "PIZZA MIT ANANAS",
            "BENUTZT COMIC SANS",
            "Socken in Sandalen",
            "Hat den Stift geklaut",
            "Zu lautes Atmen",
            "Kaffee verschüttet",
            "Spoiler verraten",
            "Drängelt vor",
            "Singt unter der Dusche",
            "Hat Cookies abgelehnt"
        ]

        threat_levels = [
            "HARMLESS",
            "LOW RISK",
            "COFFEE NEEDED",
            "EXTREME DANGER",
            "AVENGERS LEVEL",
            "RUN AWAY!"
        ]

        return {
            "age": str(random.randint(18, 99)),
            "gender": random.choice(genders),
            "emotion": random.choice(emotions),
            "clothing": random.choice(clothes),
            # Neue Werte generieren:
            "height": f"{random.randint(155, 205)} cm",
            "threat": random.choice(threat_levels),
            "crime": random.choice(crimes).upper()  # .upper() macht alles GROSS
        }

    def update_loop(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            analyzed_data = None

            for (x, y, w, h) in faces:
                cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if time.time() - self.last_analysis_time > 2.0:  # Alle 2 sekunden Update
                    analyzed_data = self.mock_ai_analysis()
                    self.last_analysis_time = time.time()

            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            if analyzed_data:
                self.update_report(analyzed_data)

        self.root.after(20, self.update_loop)

    def update_report(self, data):
        # Bestehende Felder updaten
        self.report_canvas.itemconfigure(self.text_ids["age"], text=f"{data['age']}")
        self.report_canvas.itemconfigure(self.text_ids["gender"], text=f"{data['gender']}")
        self.report_canvas.itemconfigure(self.text_ids["emotion"], text=f"{data['emotion']}")
        self.report_canvas.itemconfigure(self.text_ids["clothing"], text=f"{data['clothing']}")

        # --- NEUE FELDER UPDATEN ---
        self.report_canvas.itemconfigure(self.text_ids["height"], text=f"{data['height']}")
        self.report_canvas.itemconfigure(self.text_ids["threat"], text=f"{data['threat']}")
        self.report_canvas.itemconfigure(self.text_ids["crime"], text=f"{data['crime']}")

    def show_mouse_coords(self, event):
        x = event.x
        y = event.y
        self.mouse_label.config(text=f"X: {x} | Y: {y}")


if __name__ == "__main__":
    if not os.path.exists(BACKGROUND_IMAGE_PATH):
        os.makedirs(os.path.dirname(BACKGROUND_IMAGE_PATH), exist_ok=True)
        img = Image.new('RGB', (REPORT_WIDTH, REPORT_HEIGHT), color=(210, 200, 180))
        img.save(BACKGROUND_IMAGE_PATH)

    root = tk.Tk()
    app = SecurityProfilingApp(root)
    root.mainloop()