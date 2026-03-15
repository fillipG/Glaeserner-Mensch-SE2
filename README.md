# Glaeserner-Mensch-SE2
Camera recognition of different human characteristics such as gender, age or mood
 
---
 
# Anleitung Programm starten
 
### 0. Schriftarten installieren
 
Fuer die korrekte Darstellung der GUI werden zwei Schriftarten benoetigt.
Beide muessen einmalig auf dem System installiert werden:
 
* **Graduate**: https://fonts.google.com/specimen/Graduate
* **Goudy Bookletter 1911**: https://fonts.google.com/specimen/Goudy+Bookletter+1911
 
Schriftart installieren:
1. Link oeffnen und auf **Get font** klicken, dann auf **Download all** klicken
2. ZIP-Datei entpacken
3. `.ttf`-Datei(en) per Doppelklick oeffnen und auf **Installieren** klicken
 
Ohne diese Schriftarten wird die GUI nicht korrekt dargestellt.
 
---
 
### 1. Lokale Python-Umgebung vorbereiten
 
* Fuer dieses Projekt wird **Python 3.10** empfohlen: https://www.python.org/downloads/
* Die lokalen Python-Pakete im Projektordner installieren:
 
```powershell
py -3.10 -m pip install -r requirements.txt
```
 
* `torch` separat installieren, damit die passende CPU- oder GPU-Version bewusst gewaehlt wird.
  Den passenden Befehl fuer dein System findest du hier:
  `https://pytorch.org/get-started/locally/`
 
* Falls du eine NVIDIA-GPU verwendest, ist zum Beispiel dieser Weg moeglich:
 
```powershell
py -3.10 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
 
* Falls keine NVIDIA-GPU genutzt wird:
 
```powershell
py -3.10 -m pip install torch torchvision torchaudio
```
 
---
 
### 2. Ollama installieren und Modell laden
 
* Ollama von `https://ollama.com/download` installieren.
* Danach das im Projekt verwendete Standardmodell laden:
 
```powershell
ollama pull qwen2.5:3b
```
 
* `ollama serve` muss auf Windows im Normalfall **nicht** manuell gestartet werden.
  Die App startet den lokalen Ollama-Dienst bei Bedarf selbst, sofern `ollama` im PATH gefunden wird.
 
---
 
### 3. Docker Desktop starten
 
* Docker Desktop oeffnen und warten, bis der Dienst komplett gestartet ist.
 
---
 
### 4. Docker-Container starten
 
Im Projekt werden mehrere Compose-Dateien verwendet.
 
**Zuerst Root-Compose (Moondream):**
 
```powershell
docker compose build
docker compose up -d
```
 
**Dann Face-YOLO:**
 
```powershell
cd "General ordner/docker-compose-face-Yolo"
docker compose build
docker compose up -d
cd ../..
```
 
**Dann DeepFace:**
 
```powershell
cd "General ordner/docker-compose-deepface"
docker compose build
docker compose up -d
cd ../..
```
 
* FER ist optional und muss nur gestartet werden, wenn es in der `config.yaml` aktiviert werden soll.
* Die Container laufen danach im Hintergrund und koennen ueber Docker Desktop oder ueber `docker compose stop` in den jeweiligen Compose-Ordnern beendet werden.
 
---
 
### 5. main.py starten
 
Die Anwendung immer ueber ein Terminal starten:
 
```powershell
py -3.10 main.py
```
 
**Wichtig:**
* Nicht `python main.py` verwenden.
* Auf manchen Windows-Systemen zeigt `python` auf einen fehlerhaften Windows-Store-Alias.
* `py -3.10 main.py` ist fuer dieses Projekt der sichere Startweg.
 
---
 
### 6. Wichtige Hinweise zum Lauf
 
* Sowohl die lokale Python-Umgebung als auch die Docker-Container muessen korrekt laufen.
* Beim Start werden alte Zwischenordner und Inboxen automatisch geleert, damit das System in einem sauberen Zustand beginnt.
* Der lokale Ollama-Worker wird von der App gestartet und verarbeitet Textausgaben fuer die Pipeline.
* Ergebnisse der finalen Modelle landen in `./General ordner/final`.
* Wenn Fehlermeldungen wichtig sind, die App immer ueber ein Terminal starten und nicht per Doppelklick auf `main.py`.
* **Auto-Close:** Die App schliesst die angezeigte Akte automatisch, wenn keine Person mehr vor der Kamera erkannt wird. Das ist kein Absturz, sondern eine eingebaute Funktion. Im Admin-Menue deaktivierbar.
