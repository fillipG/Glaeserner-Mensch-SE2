# Glaeserner-Mensch-SE2
Camera recognition of different human characteristics such as height, age or mood


# Anleitung Modell hinzufügen

---

### 1. Einen neuen Modell-Worker hinzufügen
1. **Ordner erstellen:** Erstelle ein neues Verzeichnis für dein Modell (z. B. `./face_rec_ai`).
2. **Worker-Skript:** Dein Python-Skript muss:
    * Den Ordner `./faces_yolo` nach neuen Bildern überwachen.
    * Die Ergebnisse in den Ordner `./final` schreiben.
    * **Dateinamen-Regel:** Die Datei muss so benannt werden: `{faceID}_{modelID}.yaml`  
      *(Beispiel: `face1_face_rec.yaml`)*
    * **YAML-Inhalt:** Die Datei muss den Output des Modells mit Bezeichnung enthalten. Beispiel `{Beschreibung}: {Output}`  

---

### 2. config.yaml aktualisieren
Trage dein Modell in die `pipeline`-Liste der `config.yaml` ein. 
* Nutze die dort hinterlegten Kommentare als Vorlage.
* **WICHTIG:** Die `id` muss exakt mit dem `{modelID}`-Teil deines Dateinamens übereinstimmen.

---

### 3. Modell in docker-compose.yml hinzufügen
Damit dein Modell automatisch mit dem System startet und Zugriff auf die Dateien hat, füge es unter `services:` hinzu.

---
# Anleitung Programm starten

### 1. Befehl `docker-compose up -d --build` ausführen
* Die Container laufen jetzt im Hintergrund und können über Docker Desktop oder den Befehl `docker-compose stop` beendet werden
---
### 2. main.py starten
* Docker braucht etwas länger und zeigt nicht an, sobald es fertig ist
* Der Ordner `./faces_yolo` wird, sobald Docker fertig ist, von moondream (und idealerweise eurem Modell) überwacht
* Ergebnisse werden in `./final` geschrieben
* Änderungen werden im IDE oft verzögert angezeigt, also besser im Explorer nachschauen
* main.py überwacht `./final` und gibt die Ergebnisse im Terminal aus, sobald alle Modelle fertig sind