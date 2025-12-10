# Bitte folgende Befehle ausführen
# pip install opencv-python pillow

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import random  # Nur für die Simulation der KI-Daten


class AIProfilerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CRIMINAL PROFILING REPORT")
        self.root.geometry("600x700")

        # --- Design-Konfiguration (Akten-Look) ---
        self.colors = {
            "folder_bg": "#8B5A2B",  # Dunkles Braun (Aktenmappe)
            "paper_bg": "#F5E6C6",  # Helles Beige (Altes Papier)
            "text_main": "#2F2F2F",  # Dunkelgrau (Schreibmaschinentext)
            "accent": "#8B0000"  # Dunkelrot (Stempel/Wichtig)
        }

        self.fonts = {
            "header": ("Courier New", 24, "bold", "underline"),
            "label": ("Courier New", 14, "bold"),
            "value": ("Courier New", 14)
        }

        # Hauptfenster konfigurieren
        self.root.configure(bg=self.colors["folder_bg"])

        # --- Variablen für die KI-Daten ---
        self.var_alter = tk.StringVar(value="Analysiere...")
        self.var_groesse = tk.StringVar(value="Analysiere...")
        self.var_emotion = tk.StringVar(value="Analysiere...")
        self.var_kleidung = tk.StringVar(value="Analysiere...")
        self.var_geschlecht = tk.StringVar(value="Analysiere...")

        # --- GUI Aufbau: Die Papierseite auf der Akte ---
        self.create_paper_ui()

        # --- Zweites Fenster für die Kamera öffnen ---
        self.open_camera_window()

        # --- Simulation starten (Später durch echte KI ersetzen) ---
        self.simulate_ai_updates()

    def create_paper_ui(self):
        """Erstellt das Blatt Papier auf dem Hintergrund der Akte"""

        # Ein Rahmen, der wie ein Blatt Papier aussieht (mit Randabstand)
        paper_frame = tk.Frame(self.root, bg=self.colors["paper_bg"], bd=10, relief="flat")
        paper_frame.pack(expand=True, fill="both", padx=20, pady=20)

        # Überschrift
        lbl_header = tk.Label(paper_frame, text="CRIMINAL PROFILING REPORT",
                              font=self.fonts["header"], bg=self.colors["paper_bg"], fg=self.colors["text_main"])
        lbl_header.pack(pady=(30, 40))

        # Platzhalter für ein Profilbild (Optional, falls die KI Gesichter ausschneidet)
        # Hier zeichnen wir einen leeren Rahmen
        self.profile_canvas = tk.Canvas(paper_frame, width=150, height=150, bg="#e0e0e0", highlightthickness=1,
                                        highlightbackground="black")
        self.profile_canvas.create_text(75, 75, text="SUBJECT", font=("Courier New", 10))
        self.profile_canvas.pack(pady=(0, 30))

        # Daten-Bereich (Grid Layout für saubere Ausrichtung)
        data_frame = tk.Frame(paper_frame, bg=self.colors["paper_bg"])
        data_frame.pack(pady=10)

        # Helper Funktion zum Erstellen der Zeilen
        def create_row(label_text, variable, row_idx):
            tk.Label(data_frame, text=label_text.upper() + ":", font=self.fonts["label"],
                     bg=self.colors["paper_bg"], fg=self.colors["text_main"], anchor="e").grid(row=row_idx, column=0,
                                                                                               padx=10, pady=10,
                                                                                               sticky="e")

            tk.Label(data_frame, textvariable=variable, font=self.fonts["value"],
                     bg=self.colors["paper_bg"], fg="black", anchor="w").grid(row=row_idx, column=1, padx=10, pady=10,
                                                                              sticky="w")

        # Felder erstellen
        create_row("Alter", self.var_alter, 0)
        create_row("Geschlecht", self.var_geschlecht, 1)
        create_row("Größe", self.var_groesse, 2)
        create_row("Emotion", self.var_emotion, 3)
        create_row("Kleidungsstil", self.var_kleidung, 4)

        # Ein "Stempel" unten rechts
        lbl_stamp = tk.Label(paper_frame, text="CONFIDENTIAL\nCASE NO. 734-B",
                             font=("Impact", 16), fg=self.colors["accent"], bg=self.colors["paper_bg"],
                             borderwidth=2, relief="solid", padx=10, pady=5)
        lbl_stamp.place(relx=0.95, rely=0.95, anchor="se")
        # Leicht rotieren ist in Tkinter schwer, wir lassen es gerade.

    def open_camera_window(self):
        """Öffnet ein separates Fenster für den Kamera-Feed"""
        self.cam_window = tk.Toplevel(self.root)
        self.cam_window.title("Live Überwachung")
        self.cam_window.geometry("640x520")
        self.cam_window.configure(bg="black")

        self.video_label = tk.Label(self.cam_window, bg="black")
        self.video_label.pack(expand=True, fill="both")

        # Kamera initialisieren
        self.cap = cv2.VideoCapture(0)  # 0 ist meistens die Standard-Webcam

        self.update_camera()

    def update_camera(self):
        """Liest Frames von der Webcam und zeigt sie im GUI an"""
        ret, frame = self.cap.read()
        if ret:
            # OpenCV nutzt BGR, Tkinter braucht RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # (Optional) Hier später KI-Visualisierungen ins Bild zeichnen (Bounding Boxes etc.)
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        # Funktion alle 10ms erneut aufrufen
        self.cam_window.after(10, self.update_camera)

    def update_profile_data(self, alter, geschlecht, groesse, emotion, kleidung):
        """Diese Funktion wird später von deiner KI aufgerufen"""
        self.var_alter.set(f"{alter} Jahre")
        self.var_geschlecht.set(geschlecht)
        self.var_groesse.set(f"{groesse} cm")
        self.var_emotion.set(emotion)
        self.var_kleidung.set(kleidung)

    def simulate_ai_updates(self):
        """Simuliert eingehende Daten (NUR ZUM TESTEN)"""
        # Zufallsdaten generieren
        alter = random.randint(20, 60)
        geschlecht = random.choice(["Männlich", "Weiblich", "Divers"])
        groesse = random.randint(160, 195)
        emotion = random.choice(["Neutral", "Wütend", "Glücklich", "Verängstigt", "Überrascht"])
        kleidung = random.choice(["Casual (T-Shirt)", "Formal (Anzug)", "Sportlich", "Winterkleidung"])

        # Update durchführen
        self.update_profile_data(alter, geschlecht, groesse, emotion, kleidung)

        # Alle 3 Sekunden neue "Erkenntnis" simulieren
        self.root.after(3000, self.simulate_ai_updates)

    def on_closing(self):
        """Aufräumen beim Schließen"""
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


# --- Hauptprogramm ---
if __name__ == "__main__":
    root = tk.Tk()
    app = AIProfilerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()