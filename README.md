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
 
* Fuer dieses Projekt ist **Python 3.10 vorausgesetzt**. Das ist der aktuell einzige getestete Weg: https://www.python.org/downloads/
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
 
Die produktiven KI-Container werden zentral ueber die Root-Compose gestartet.
Damit laufen `moondream`, `face-yolo` und `deepface` gemeinsam an.

Wichtig: Der aktuelle Docker-Betrieb setzt eine funktionierende **NVIDIA-GPU-Unterstuetzung in Docker Desktop** voraus.
Die Container koennen sonst zwar starten, die Modelle laufen dann aber nicht korrekt.
Moondream erkennt automatisch ob CUDA verfuegbar ist. Ohne NVIDIA-GPU
laeuft Moondream automatisch auf CPU - langsamer aber stabil. Face-YOLO
und DeepFace benoetigen weiterhin eine NVIDIA-GPU.
 
```powershell
docker compose -f compose.yaml build
docker compose -f compose.yaml up -d
```
 
Optional zur Pruefung:
 
```powershell
docker compose -f compose.yaml ps
```
 
* Die Container laufen danach im Hintergrund und koennen ueber Docker Desktop oder ueber `docker compose -f compose.yaml stop` im Projekt-Root beendet werden.
* Die aelteren Compose-Dateien in den Unterordnern bleiben vorerst als Fallback im Repository, sind aber nicht mehr der empfohlene Startweg.
* Beim **ersten** Start kann `moondream` mehrere Minuten im Zustand `startup` bleiben, weil Modell- und Cache-Daten initial geladen werden. Das ist beim Erstlauf normal und kein Fehler.

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
* Ergebnisse der finalen Modelle landen in `./general_ordner/final`.
* Wenn Fehlermeldungen wichtig sind, die App immer ueber ein Terminal starten und nicht per Doppelklick auf `main.py`.
* **Auto-Close:** Die App schliesst die angezeigte Akte automatisch, wenn keine Person mehr vor der Kamera erkannt wird. Das ist kein Absturz, sondern eine eingebaute Funktion. Im Admin-Menue deaktivierbar.

---

### 7. Betrieb und Diagnose

Im normalen Betrieb muessen die Docker-Container nicht bei jedem einzelnen App-Start neu gebaut werden.
Wenn der Rechner bereits laeuft und die Container noch aktiv sind, reicht es in der Regel, nur die Anwendung ueber `py -3.10 main.py` zu starten.

Wenn der Rechner neu gestartet wurde oder die Container gestoppt sind, ist der uebliche Ablauf:

```powershell
docker compose -f compose.yaml up -d
py -3.10 main.py
```

Nach Betriebsschluss koennen die Container bei Bedarf wieder gestoppt werden:

```powershell
docker compose -f compose.yaml stop
```

Wenn Docker-Images aktualisiert oder nach Aenderungen neu gebaut werden sollen:

```powershell
docker compose -f compose.yaml up -d --build
```

Zur Laufzeit schreiben die drei Docker-Worker Statusdateien nach:

* `general_ordner/status/face-yolo/heartbeat.json`
* `general_ordner/status/deepface/heartbeat.json`
* `general_ordner/status/moondream/heartbeat.json`

Diese Heartbeat-Dateien enthalten Zeitstempel, Status und Prozess-ID.
Moegliche Statuswerte sind:

* `startup`
* `ready`
* `idle`
* `processing`
* `error`

Docker Desktop und `docker compose -f compose.yaml ps` zeigen auf Basis dieser Heartbeats den Health-Status der Container an:

* `healthy`: Der Worker schreibt regelmaessig und gilt als aktiv.
* `unhealthy`: Der Heartbeat ist zu alt oder der Worker hat einen Fehlerzustand gemeldet.

Wichtig:

* `unhealthy` allein startet den Container nicht neu.
* Der eigentliche Neustart erfolgt ueber den internen Watchdog des Workers zusammen mit `restart: unless-stopped`.

Wenn ein Container laenger `unhealthy` bleibt, zuerst die Logs pruefen:

```powershell
docker compose -f compose.yaml logs --tail 100 face-yolo
docker compose -f compose.yaml logs --tail 100 deepface
docker compose -f compose.yaml logs --tail 100 moondream
```

Fehlerhafte Eingabedateien werden nicht mehr still geloescht, sondern pro Service in einen `failed/`-Ordner verschoben:

* `general_ordner/moondream_ai/failed`
* `general_ordner/docker-compose-deepface/failed`

Zu jeder fehlgeschlagenen Datei wird eine `*_error.yaml` geschrieben.
Diese enthaelt unter anderem Zeitstempel, Quelldatei, Fehlergrund und den Namen der verschobenen Datei.

Die `failed/`-Ordner werden automatisch bereinigt. Dateien aelter als 14 Tage werden geloescht.
Einzelne Eintraege in `failed/` sind noch kein automatischer Alarmfall. Wenn dort aber regelmaessig neue Dateien auftauchen, sollte der jeweilige Docker-Service genauer geprueft werden.

---

### 8. Moondream-Revision bewusst aktualisieren

Moondream laeuft absichtlich auf einer gepinnten Revision. Dadurch aendert sich das Modell nicht unbemerkt zwischen zwei Betriebstagen.

Die aktuell verwendete Revision steht in `compose.yaml` in `MOONDREAM_REVISION`.

Empfohlener Weg fuer ein bewusstes Update:

1. Revision in `compose.yaml` aendern.
2. Moondream neu bauen und starten:

```powershell
docker compose -f compose.yaml up -d --build moondream
```

3. Danach einen echten Testdurchlauf mit `py -3.10 main.py` machen.
4. Erst nach erfolgreichem Test die neue Revision im Museumsbetrieb verwenden.
