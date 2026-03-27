# Datenfluss - Glaeserner Mensch

Vollstaendige Beschreibung des aktuellen Programmdurchlaufs von Start bis Mappe-Schliessen.

---

## Uebersicht

```text
Programmstart
    |
    v
Kamera-Scan (IDLE) <- Live-Preview in GUI
    |
    |  Person erkannt + Countdown abgelaufen
    v
Foto ausloesen -> General ordner/main_image/face_trigger.jpg
    |
    v
Docker: Face-YOLO
    |-- schreibt General ordner/final/faces_log.yaml
    |-- schreibt General ordner/sketch/batch123_face1.png
    |-- schreibt General ordner/docker-compose-deepface/deepface_inbox/batch123_face1.png
    `-- schreibt General ordner/moondream_ai/moondream_inbox/batch123_face1.png
    |
    |--> Docker: DeepFace  -> final/batch123_face1_deepface.yaml
    |--> Docker: Moondream -> ollama_inbox/batch123_face1_ollama.yaml
    `--> Lokal: Ollama     -> final/batch123_face1_ollama.yaml
    |
    v
PipelineManager scannt final/
    |
    |  Batch vollstaendig
    v
GUI uebernimmt fertigen Batch direkt aus der Pipeline
    |
    v
Oeffnungs-Animation
    |
    v
Mappe offen - Personen-Akten angezeigt
    |
    |-- Auto-Close
    |-- Reset
    `-- Pipeline-Timeout
    |
    v
Schliess-Animation
    |
    |-- Kamera-Prewarm im Hintergrund
    v
Geschlossener Ordner - Kamera laeuft wieder
```

---

## Phase 1 - Programmstart

**Einstiegspunkt:** `py -3.10 main.py`

### 1.1 YOLO-Modell vorbereiten

```text
main.py: run_app()
    `-- YOLOWorker.prepare()
            |-- ultralytics YOLO("yolov8n-pose.pt") laden
            |-- PersonPhotoCapture(model, photo_delay) initialisieren
            `-- os.makedirs(path_service["main_image"])
```

Das Modell wird bewusst vor dem GUI-Start geladen, damit Ladefehler sofort sichtbar sind.

### 1.2 GUI starten

```text
main.py
    `-- ScalingAkteGUI()
            |-- Zustandsvariablen, Timer, Sounds initialisieren
            |-- ConfigService laden
            `-- show_closed_folder() -> geschlossenen Ordner aufbauen
```

### 1.3 Startup-Cleanup

```text
main.py
    `-- path_service.get_paths() -> Ordner leeren:
            |-- General ordner/final/
            |-- General ordner/main_image/
            |-- General ordner/sketch/
            |-- General ordner/ollama_ai/ollama_inbox/
            |-- General ordner/docker-compose-deepface/deepface_inbox/
            `-- General ordner/moondream_ai/moondream_inbox/
```

Ohne dieses Cleanup koennten alte KI-Ergebnisse beim Neustart als neue Daten erkannt werden.

### 1.4 Lokale Worker starten

```text
main.py
    `-- LocalWorkerManager.start_ollama_worker()
            |-- is_ollama_enabled() pruefen
            |-- _ensure_ollama_ready(model_name)
            `-- subprocess.Popen("workers/ollama_worker.py")
```

### 1.5 PipelineWorker starten

```text
main.py
    `-- PipelineWorker (QThread)
            |-- PoolLoader("config.yaml") laden
            |-- PipelineManager(config_data, pool_loader) erstellen
            `-- data_finalized -> result_ready weiterleiten
```

### 1.6 YOLOWorker starten und Signale verdrahten

```text
main.py
    |-- yolo.frame_ready -> gui.on_camera_frame
    |-- yolo.person_presence_changed -> gui.on_person_presence_changed
    |-- yolo.photo_done -> gui.show_loading_indicator
    |-- gui.camera_prewarm_requested -> yolo.request_camera_prewarm
    |-- gui.folder_closed -> yolo.start_capture_mode
    `-- gui.presence_monitoring_requested -> yolo.start_presence_monitoring
```

---

## Phase 2 - Kamera-Scan (IDLE)

**Zustand:** `YOLOWorker.state = IDLE`, `GUIState.IDLE`

```text
YOLOWorker.run()  [QThread]
    `-- Endlosschleife:
            |-- config.yaml neu einlesen
            `-- PersonPhotoCapture.capture_mode()
                    |-- Kamera oeffnen (cv2.VideoCapture, Index 0/1/2)
                    |-- Frame lesen
                    |-- YOLO-Inferenz -> Personen-Keypoints
                    |-- frame_ready.emit(frame) -> GUI zeigt Live-Preview
                    |
                    |-- Person erkannt?
                    |   |-- NEIN -> naechste Runde
                    |   `-- JA -> Countdown starten
                    |           |-- Person muss stabil im Bild bleiben
                    |           |-- frame_ready weiter senden
                    |           `-- Countdown = 0 -> Frame zurueckgeben
                    |
                    `-- Captured-Frame = gespeichertes Foto
```

**GUI parallel:**

```text
on_camera_frame(frame)
    `-- CameraMixin
            |-- Haar-Cascade fuer Preview-Gesichter
            |-- LiveDeepFaceService optional
            `-- camera_pixmap_item.setPixmap()
```

---

## Phase 3 - Foto ausloesen

```text
YOLOWorker (nach Countdown)
    |-- cv2.imwrite("General ordner/main_image/face_trigger.jpg", frame)
    |-- photo_capture.release_camera()
    |-- start_analyzing_mode() -> state = ANALYZING
    `-- photo_done.emit() -> GUI: show_loading_indicator()
```

**Wichtig:** Im `ANALYZING`-Zustand bleibt die Kamera normalerweise geschlossen. Beim spaeteren Schliessen der Mappe kann sie aber bereits im Hintergrund wieder vorgewaermt werden.

---

## Phase 4 - KI-Analyse

Der Analyse-Teil besteht aus zwei Stufen:

1. Face-YOLO zerlegt das Foto in Gesichter und verteilt die Dateien.
2. DeepFace, Moondream und Ollama verarbeiten diese Dateien weiter.

### 4.1 Face-YOLO

Der Face-YOLO-Container (`docker-compose-face-Yolo/face_yolo.py`) ist die zentrale Verarbeitungsstufe. Er uebernimmt Gesichtserkennung, optionale Koerpererkennung und die Verteilung der Crops an die nachgelagerten KI-Dienste.

**Alle Parameter werden vor jedem Bild live aus `config.yaml` geladen.**

```text
Face-YOLO  [Endlosschleife, 0.5s Takt]
    |
    |-- Liest: General ordner/main_image/face_trigger.jpg
    |-- build_batch_id() -> z.B. "batch1234567890"
    |
    |-- cleanup_previous_batch()
    |   |-- loescht alte face*- und batch*_face*-Dateien in sketch/
    |   |-- loescht alte face*- und batch*_face*-Dateien in deepface_inbox/
    |   |-- loescht alte face*- und batch*_face*-Dateien in moondream_inbox/
    |   |-- loescht alte face*_*.yaml und batch*_face*_*.yaml in final/
    |   `-- loescht alte Debug-Dateien in debug_body/
    |
    |-- preprocess_for_detection()
    |   |-- Aufhellung
    |   |-- CLAHE-Kontrast
    |   `-- Scharfzeichnen
    |
    |-- yolov8n-face.pt -> Gesichtserkennung
    |-- select_best_face_candidates()
    |   |-- 45% Konfidenz
    |   |-- 30% Flaeche
    |   |-- 15% Bildmitte
    |   `-- 10% Schaerfe
    |
    |-- sortiert Treffer von links nach rechts
    |-- schreibt faces_log.yaml mit batch_id und face_count
    |
    `-- falls Modus != face:
            |-- yolov8n.pt -> Personenerkennung (classes=[0])
            `-- match_faces_to_persons()
```

**Pro erkanntem Gesicht (`batch123_faceN`):**

```text
    |-- face_crop + rembg
    |   |-- sketch/batch123_faceN.png
    |   `-- deepface_inbox/batch123_faceN.png
    |
    |-- build_moondream_crop()
    |   |
    |   |-- "face"
    |   |   `-- freigestellter Face-Crop
    |   |
    |   |-- "body"
    |   |   |-- build_body_crop()
    |   |   |-- nutzt Personenhoehe
    |   |   |-- zentriert horizontal um das Gesicht
    |   |   |-- body_padding_ratio beeinflusst den Rand
    |   |   `-- Fallback auf Face wenn erlaubt
    |   |
    |   |-- "body_seg"
    |   |   |-- yolov8n-seg.pt
    |   |   |-- isoliert die Person pixelgenau
    |   |   |-- Hintergrund wird weiss gesetzt
    |   |   `-- Fallback auf Face wenn erlaubt
    |   |
    |   `-- "shadow"
    |       |-- Moondream bekommt face_no_bg
    |       `-- body/body_seg gehen nur nach debug_body/
    |
    |-- moondream_inbox/batch123_faceN.png
    `-- loescht main_image/face_trigger.jpg nach dem letzten Gesicht
```

**Debug-Ausgaben** (bei Body-/Segmentation-Pfaden und im Shadow-Debug):

```text
General ordner/debug_body/
    |-- batch123_faceN_moondream.png
    |-- batch123_faceN_body_seg.png
    |-- batch123_faceN_body.png
    `-- matching_debug.yaml
```

**Konfigurationsparameter (`face_yolo:` in `config.yaml`):**

| Parameter | Default | Effekt |
|---|---|---|
| `confidence` | 0.5 | Schwellenwert fuer `yolov8n-face.pt` |
| `max_faces` | 4 | Maximal verarbeitete Gesichter pro Foto |
| `moondream_crop_mode` | `face` | `face`, `body`, `body_seg`, `shadow` |
| `body_confidence` | 0.35 | Schwellenwert fuer `yolov8n.pt` und `yolov8n-seg.pt` |
| `body_padding_ratio` | 0.12 | Zusaetzlicher Rand fuer Body- und Seg-Crops |
| `body_fallback_to_face` | true | Bei fehlendem brauchbarem Body-Crop auf Face zurueckfallen |
| `body_matching_required` | false | Fehlendes Match als harteren Fehler behandeln |
| `debug_matching` | true | `matching_debug.yaml` schreiben |

### 4.2 DeepFace

```text
DeepFace
    |-- liest deepface_inbox/batch123_faceN.png
    |-- erkennt Alter, Geschlecht, dominante Emotion
    |-- schreibt final/batch123_faceN_deepface.yaml
    `-- loescht deepface_inbox/batch123_faceN.png
```

### 4.3 Moondream

```text
Moondream
    |-- liest moondream_inbox/batch123_faceN.png
    |-- erstellt visuelle Beschreibung
    |-- schreibt ollama_inbox/batch123_faceN_ollama.yaml
    `-- loescht moondream_inbox/batch123_faceN.png
```

### 4.4 Ollama-Worker

```text
workers/ollama_worker.py
    |-- scannt ollama_inbox/ nach batch*_face*_ollama.yaml
    |-- liest moondream_description + moondream_prompt
    |-- baut Prompt
    |-- ruft ollama.chat(...) auf
    |-- schreibt final/batch123_faceN_ollama.yaml
    `-- loescht die Inbox-Datei
```

**Sonderfall Ollama deaktiviert:**

```text
process_file_passthrough()
    `-- schreibt final/batch123_faceN_moondream.yaml
```

---

## Phase 5 - PipelineManager ueberwacht `final/`

```text
PipelineWorker.run()  [QThread, 0.5s Takt]
    `-- PipelineManager.check_for_updates()
            |
            |-- Schritt 1: faces_log.yaml lesen
            |   |-- Signatur = (mtime, batch_id, face_count)
            |   |-- neue Signatur -> Cache, Batch-Zustand und Sammellisten leeren
            |   |-- face_count == 0 -> data_finalized.emit("EMPTY", [])
            |   `-- current_batch_id setzen
            |
            |-- Schritt 2: neue .yaml-Dateien in final/ scannen
            |   |-- batch123_face1_deepface.yaml -> base_id="batch123_face1"
            |   |-- batch123_face1_ollama.yaml   -> base_id="batch123_face1"
            |   `-- Dateien anderer Batches werden ignoriert
            |
            |-- Schritt 3: Vollstaendigkeit pruefen
            |   |-- required_ids = ["ollama", "deepface"]
            |   `-- wenn komplett -> add_to_batch(base_id)
            |
            |-- Schritt 4: Person-Dict bauen
            |   |-- Beschreibung aus Ollama oder Moondream
            |   |-- DeepFace-Daten dazumischen
            |   |-- Gefahr aus Emotion ableiten
            |   `-- Sketch-Pfad fuer GUI direkt mitgeben
            |
            |-- Schritt 5: Pool auffuellen
            |   |-- _append_pool_people(collected_faces)
            |   `-- Pool-Logik lebt nur hier, nicht mehr in der GUI
            |
            `-- data_finalized.emit("BATCH", data)
```

---

## Phase 6 - GUI empfaengt Batch-Signal

```text
ScalingAkteGUI.handle_pipeline_result(status, personen_daten)
    |
    |-- status == "EMPTY"
    |   `-- close_folder("empty_result")
    |
    `-- status == "BATCH"
            |-- Besucherstatistik ueber faces_log.yaml aktualisieren
            |-- _auto_close_monitoring_pending = True
            `-- handle_new_dataset(personen_daten)
                    |-- fertigen Batch direkt uebernehmen
                    |-- Mappe offen?
                    |   `-- JA -> Flip-Animation
                    `-- Mappe geschlossen?
                        `-- Oeffnungs-Animation
```

Die GUI baut den Pool nicht mehr selbst und rekonstruiert den Batch nicht mehr aus `final/`.

---

## Phase 7 - Mappe oeffnen

```text
show_animation_with_timer()
    |-- Developer-Modus? -> Animation sofort
    `-- sonst Countdown-Kreis starten

start_animation(open_animation.mp4)
    |-- GUIState -> OPENING
    |-- hide_loading_indicator()
    `-- update_video_frame() per QTimer
            `-- Video fertig -> show_open_folder()
```

---

## Phase 8 - Mappe offen

```text
show_open_folder()
    |-- GUIState -> RESULTS_READY
    |-- Sound: folder_open
    |-- setup_ui_elements()
    |   |-- PersonContainer x4
    |   |-- Sketch/Bild aus mitgegebenem Batch oder Pool
    |   `-- Buttons aufbauen
    |-- update_descriptions_from_files()
    |   `-- liest aktuelle Batch-Dateien bei Bedarf nochmals aus final/
    `-- falls Auto-Close aktiv:
            presence_monitoring_requested.emit()
            -> YOLOWorker.start_presence_monitoring()
```

### 8.1 Typewriter-Animation

```text
PersonContainer.trigger_typing()
    `-- UiTypewriter.start_typing()
```

### 8.2 Auto-Close-Ueberwachung

```text
YOLOWorker  [PRESENCE_MONITORING]
    `-- PersonPhotoCapture.presence_mode()
            |-- Frame lesen
            |-- YOLO-Inferenz: Person vorhanden?
            `-- person_presence_changed.emit(True/False)

GUI: on_person_presence_changed(is_present)
    |-- True  -> Zaehler und Warnung zuruecksetzen
    `-- False -> _missed_presence_checks += 1
            |-- Warnschwelle erreicht? -> Reset-Button blinkt
            `-- Limit erreicht? -> close_folder("auto_close")
```

### 8.3 Reset

```text
btn_reset.clicked -> reset_logic()
    `-- Countdown oder direktes close_folder("manual"/"manual_countdown")
```

### 8.4 Sprache wechseln

```text
btn_language.clicked -> switch_language_logic()
    |-- Zielsprache wechseln
    |-- Container aktualisieren
    `-- Sprache in config.yaml speichern
```

---

## Phase 9 - Mappe schliessen

```text
close_folder(reason)
    |-- camera_prewarm_requested.emit()
    |   `-- YOLOWorker.request_camera_prewarm()
    |-- _clear_pipeline_outputs_on_close setzen
    |-- falls Animation laeuft: _pending_close merken
    `-- sonst close_animation.mp4 starten
```

```text
YOLOWorker  [ANALYZING oder PRESENCE_MONITORING]
    `-- request_camera_prewarm()
            |-- Kamera im Worker-Thread oeffnen
            |-- Test-Frame lesen
            `-- Kamera bis zum Ruecksprung nach IDLE offen halten
```

```text
show_closed_folder()
    |-- Pipeline-Ausgaben bei Bedarf leeren
    |-- _is_open = False
    |-- GUIState -> IDLE
    |-- Szene neu aufbauen
    |-- schwarzer Kamera-Platzhalter
    `-- folder_closed.emit()
            -> YOLOWorker.start_capture_mode()
                    `-- state = IDLE; Kamera sollte bereits vorgewaermt sein
```

---

## Vollstaendiger Kreislauf

```text
close_folder()
    |-- camera_prewarm_requested.emit() -> YOLOWorker.request_camera_prewarm()
    `-- ... Schliess-Animation ...
            |
            v
      show_closed_folder()
          `-- folder_closed.emit() -> YOLOWorker.start_capture_mode()
                                          |
                                          v
                                <- Phase 2: Kamera-Scan (IDLE) <-
```

---

## Dateien und ihre Rollen

| Datei / Ordner | Rolle |
|---|---|
| `config.yaml` | Zentrale Konfiguration |
| `path_service.py` | Zentrale Pfadauflosung |
| `General ordner/main_image/face_trigger.jpg` | Kamera-Foto als Eingang fuer Face-YOLO |
| `General ordner/final/faces_log.yaml` | Aktueller `batch_id`, `face_count` und weitere Batch-Metadaten |
| `General ordner/final/batch123_face1_deepface.yaml` | Ergebnis von DeepFace |
| `General ordner/ollama_ai/ollama_inbox/batch123_face1_ollama.yaml` | Moondream-Zwischenergebnis fuer Ollama |
| `General ordner/final/batch123_face1_ollama.yaml` | Finales Ollama-Ergebnis |
| `General ordner/final/batch123_face1_moondream.yaml` | Fallback-Ausgabe, wenn Ollama deaktiviert ist |
| `General ordner/sketch/batch123_face1.png` | Von Face-YOLO erzeugter GUI-Crop |
| `General ordner/debug_body/` | Debug-Ausgaben fuer Body-/Segmentation-Pfade |
| `./pool/` | Vorbefuellte Pool-Personen |
| `translation_cache.yaml` | Lokaler Uebersetzungs-Cache |

---

## Threads und Prozesse

| Komponente | Typ | Kommunikation |
|---|---|---|
| Qt-Eventschleife + GUI | Hauptthread | - |
| `YOLOWorker` | QThread | Signale: `frame_ready`, `photo_done`, `person_presence_changed`; Methoden: `start_capture_mode()`, `start_presence_monitoring()`, `request_camera_prewarm()` |
| `PipelineWorker` | QThread | Signal: `result_ready` -> `handle_pipeline_result` |
| `ollama_worker.py` | Subprozess | Liest `ollama_inbox/`, schreibt `final/` |
| Face-YOLO-Container | Docker | Liest `main_image/`, schreibt `sketch/`, `deepface_inbox/`, `moondream_inbox/`, `final/faces_log.yaml`, `debug_body/` |
| Moondream-Container | Docker | Liest `moondream_inbox/`, schreibt `ollama_inbox/` |
| DeepFace-Container | Docker | Liest `deepface_inbox/`, schreibt `final/` |

Alle GUI-Updates laufen ueber Qt-Queued-Connections bzw. Qt-Signale.

---

## Ordner-Pfade konfigurieren

Standard: `General ordner/`

```yaml
paths:
  base_dir: general_ordner
```

Alle Standardordner (`final`, `main_image`, `ollama_inbox`, `deepface_inbox`, `moondream_inbox`, `sketch_dir`, `face_yolo_weights`) folgen automatisch diesem Basisverzeichnis. Einzelne Pfade koennen zusaetzlich separat ueberschrieben werden.
