# Datenfluss — Gläserner Mensch

Vollständige Beschreibung des Programmdurchlaufs von Start bis Mappe-Schließen.

---

## Übersicht

```
Programmstart
    │
    ▼
Kamera-Scan (IDLE)          ← Live-Preview in GUI
    │
    │  Person erkannt + Countdown abgelaufen
    ▼
Foto auslösen               → General ordner/main_image/face_trigger.jpg
    │
    ▼
Docker: Face-YOLO
    ├── schreibt General ordner/final/faces_log.yaml
    ├── schreibt General ordner/sketch/face1.png
    ├── schreibt General ordner/docker-compose-deepface/deepface_inbox/face1.png
    └── schreibt General ordner/moondream_ai/moondream_inbox/face1.png
    │
    ├──► Docker: DeepFace    liest deepface_inbox/, schreibt final/face1_deepface.yaml
    ├──► Docker: Moondream   liest moondream_inbox/, schreibt ollama_inbox/face1_ollama.yaml
    └──► Lokal:  Ollama      liest ollama_inbox/, schreibt final/face1_ollama.yaml
    │
    ▼
PipelineManager scannt final/ alle 0,5s
    │
    │  Alle KI-Ergebnisse für alle Gesichter vollständig
    ▼
Batch vollständig → Signal an GUI
    │
    ▼
GUI liest Anzeige-Daten direkt aus General ordner/final
    │
    ▼
Öffnungs-Animation (open_animation.mp4)
    │
    ▼
Mappe offen — Personen-Akten angezeigt
    │
    ├── Auto-Close (Person weg)
    ├── Reset-Button (Nutzer)
    └── Pipeline-Timeout (KI zu langsam)
    │
    ▼
Schließ-Animation (close_animation.mp4)
    │
    ▼
Geschlossener Ordner — Kamera läuft wieder  (→ zurück zu Kamera-Scan)
```

---

## Phase 1 — Programmstart

**Einstiegspunkt:** `py -3.10 main.py`

### 1.1 YOLO-Modell vorbereitenieren
```
main.py: run_app()
    └── YOLOWorker.prepare()
            ├── ultralytics YOLO("yolov8n-pose.pt") laden
            ├── PersonPhotoCapture(model, photo_delay) initialisieren
            └── os.makedirs(path_service["main_image"])
```
Das Modell wird absichtlich **vor** dem GUI-Start geladen, damit der Ladevorgang
nicht die Qt-Eventschleife blockiert.

### 1.2 GUI starten
```
main.py
    └── ScalingAkteGUI()
            ├── Alle Zustandsvariablen, Timer, Sounds initialisieren
            ├── ConfigService laden (config.yaml → Standardwerte schreiben falls leer)
            └── show_closed_folder()  → Kamera-Preview-Bildschirm aufbauen
```

### 1.3 Startup-Cleanup
```
main.py
    └── path_service.get_paths() → alle 6 Ordner leeren:
            ├── General ordner/final/
            ├── General ordner/main_image/
            ├── General ordner/sketch/
            ├── General ordner/ollama_ai/ollama_inbox/
            ├── General ordner/docker-compose-deepface/deepface_inbox/
            └── General ordner/moondream_ai/moondream_inbox/
```
**Warum:** KI-Docker-Container schreiben YAML-Dateien in diese Ordner.
Ohne Cleanup würden Ergebnisse vom letzten Programmstart beim nächsten Start
fälschlich als neue Ergebnisse erkannt werden.

### 1.4 Lokale Worker starten
```
main.py
    └── LocalWorkerManager.start_ollama_worker()
            ├── is_ollama_enabled() → config.yaml prüfen
            ├── _ensure_ollama_ready(model_name)
            │       ├── Python-Paket "ollama" vorhanden?
            │       ├── API http://127.0.0.1:11434 erreichbar?
            │       │       └── Falls nicht: "ollama serve" starten, 15s warten
            │       └── Modell (z.B. "qwen2.5:3b") lokal vorhanden?
            └── subprocess.Popen("workers/ollama_worker.py")
                    → läuft als eigenständiger Python-Prozess im Hintergrund
```

### 1.5 PipelineWorker starten
```
main.py
    └── PipelineWorker (QThread)
            ├── PoolLoader("config.yaml") laden  → liest ./pool/ Verzeichnis
            ├── PipelineManager(config_data, pool_loader) erstellen
            │       ├── path_service.get_paths() → Ordnerpfade
            │       ├── reload_config() → enabled_models, watch_dir, required_ids bestimmen
            │       └── os.makedirs(watch_dir)
            └── data_finalized Signal → result_ready Signal weiterleiten
```

### 1.6 YOLOWorker starten + Signale verdrahten
```
main.py  (Signal-Verdrahtung)
    ├── yolo.frame_ready            → gui.on_camera_frame
    ├── yolo.person_presence_changed → gui.on_person_presence_changed
    ├── yolo.photo_done             → gui.show_loading_indicator
    ├── gui.folder_closed           → yolo.start_capture_mode
    └── gui.presence_monitoring_requested → yolo.start_presence_monitoring
```

---

## Phase 2 — Kamera-Scan (IDLE)

**Zustand:** `YOLOWorker.state = IDLE`, `GUIState.IDLE`

```
YOLOWorker.run()  [QThread]
    └── Endlosschleife:
            ├── config.yaml neu einlesen (photo_delay, language)
            └── PersonPhotoCapture.capture_mode()
                    ├── Kamera öffnen (cv2.VideoCapture, Index 0/1/2)
                    ├── Frame lesen
                    ├── YOLO-Inferenz → Personen-Bounding-Boxes + Keypoints
                    ├── frame_ready.emit(frame) → GUI zeigt Live-Preview
                    │
                    ├── Person erkannt?
                    │   ├── NEIN → Weiter zur nächsten Runde
                    │   └── JA  → Countdown starten (photo_delay Sekunden)
                    │               ├── Person muss PERSON_LOST_TOLERANCE (1.5s) im Frame bleiben
                    │               ├── frame_ready weiter senden → GUI-Countdown-Animation
                    │               └── Countdown = 0 → Frame zurückgeben
                    │
                    └── Captured-Frame = gespeichertes Foto
```

**GUI (parallel):**
```
on_camera_frame(frame)
    └── CameraMixin._filter_preview_faces()
            ├── LiveDeepFaceService (optional): Gesichter live analysieren
            └── camera_pixmap_item.setPixmap() → Live-Bild in Szene aktualisieren
```

---

## Phase 3 — Foto auslösen

```
YOLOWorker (nach Countdown)
    ├── cv2.imwrite("General ordner/main_image/face_trigger.jpg", frame)
    ├── start_analyzing_mode()  → state = ANALYZING (Kamera schläft: 0.1s Loop)
    └── photo_done.emit()  → GUI: show_loading_indicator()
```

**GUI:**
```
show_loading_indicator()
    └── LoadingSpinnerItem anzeigen (rotierendes Symbol)
        GUIState → LOADING
```

---

## Phase 4 — KI-Analyse (parallel in Containern / lokal)

Der Analyse-Teil besteht aus zwei Stufen:
1. Face-YOLO zerlegt das Foto in einzelne Gesichter und verteilt die Dateien.
2. DeepFace, Moondream und Ollama verarbeiten diese verteilten Dateien weiter.

### 4.1 Face-YOLO (Docker-Container)
```
Face-YOLO-Container
    ├── Liest: General ordner/main_image/face_trigger.jpg
    ├── cleanup_previous_batch()
    │   ├── löscht alte face*.png / face*.jpg in sketch/
    │   ├── löscht alte face*.png / face*.jpg in deepface_inbox/
    │   ├── löscht alte face*.png / face*.jpg in moondream_inbox/
    │   └── löscht alte face*_*.yaml in final/
    ├── YOLO-Gesichtserkennung auf face_trigger.jpg
    ├── Schreibt: General ordner/final/faces_log.yaml
    │               → { face_count: N }
    ├── Für jedes Gesicht faceN:
    │   ├── Schreibt: General ordner/sketch/faceN.png
    │   ├── Schreibt: General ordner/docker-compose-deepface/deepface_inbox/faceN.png
    │   └── Schreibt: General ordner/moondream_ai/moondream_inbox/faceN.png
    └── Löscht: General ordner/main_image/face_trigger.jpg
```

### 4.2 DeepFace (Docker-Container)
```
DeepFace-Container
    ├── Liest: General ordner/docker-compose-deepface/deepface_inbox/faceN.png
    ├── Erkennt: Alter, Geschlecht, dominante Emotion (RetinaFace optional)
    ├── Schreibt: General ordner/final/faceN_deepface.yaml
    │               → { Alter: 34, Geschlecht: "Mann", Emotion: "neutral" }
    └── Löscht: deepface_inbox/faceN.png
```

### 4.3 Moondream (Docker-Container)
```
Moondream-Container
    ├── Liest: General ordner/moondream_ai/moondream_inbox/faceN.png
    ├── VLM-Analyse (visuelle Beschreibung der Person)
    ├── Schreibt: General ordner/ollama_ai/ollama_inbox/faceN_ollama.yaml
    │               → { moondream_description: "...", moondream_prompt: "..." }
    └── Löscht: moondream_inbox/faceN.png
```

### 4.4 Ollama-Worker (lokaler Python-Subprozess)
```
workers/ollama_worker.py  [Endlosschleife, 0.5s Takt]
    ├── Scannt: General ordner/ollama_ai/ollama_inbox/ nach faceN_ollama.yaml
    ├── Findet: face1_ollama.yaml
    ├── Liest: { moondream_description: "...", moondream_prompt: "..." }
    ├── Baut Prompt: "Personenbeschreibung: ... \n <konfigurierter Prompt>"
    ├── ollama.chat(model="qwen2.5:3b", messages=[...])  → lokale API
    ├── normalize_single_paragraph(response)
    ├── Schreibt: General ordner/final/face1_ollama.yaml
    │               → { description: "...", source_description: "..." }
    └── os.remove(ollama_inbox/face1_ollama.yaml)   (Inbox-Datei löschen)
```

**Sonderfall Ollama deaktiviert:**
```
process_file_passthrough()
    └── Moondream-Beschreibung direkt als face1_moondream.yaml in final/ schreiben
        (Pipeline wartet dann auf moondream statt ollama)
```

---

## Phase 5 — PipelineManager überwacht `final/`

```
PipelineWorker.run()  [QThread, 0.5s Takt]
    └── PipelineManager.check_for_updates()

            Schritt 1: faces_log.yaml lesen
            ├── Zeitstempel + face_count als "Signatur" prüfen
            ├── Neue Signatur → results_cache, collected_faces, seen_files leeren
            │                   (neuer Foto-Vorgang erkannt)
            ├── face_count == 0 → data_finalized.emit("EMPTY", []) → Mappe bleibt zu
            └── expected_face_count = face_count

            Schritt 2: Neue .yaml-Dateien in final/ scannen
            ├── Dateiname: face1_deepface.yaml → base_id="face1", model_id="deepface"
            ├── Dateiinhalt in results_cache["face1"]["deepface"] speichern
            ├── Dateiname: face1_ollama.yaml   → base_id="face1", model_id="ollama"
            └── Dateiinhalt in results_cache["face1"]["ollama"] speichern

            Schritt 3: Vollständigkeitsprüfung
            ├── required_ids = ["ollama", "deepface"]  (oder ["moondream", "deepface"])
            ├── Alle required_ids für face1 vorhanden?
            └── JA → add_to_batch("face1")

            Schritt 4: Batch abschließen
            ├── _build_person_dict(base_id, df_data, description_data)
            │       ├── beschreibung = ollama.description oder moondream.description
            │       ├── Falls Sprache "de": TranslationService.translate_text(beschreibung)
            │       │       → Cache prüfen → Google Translate API → Fallback: Originaltext
            │       ├── gefahr = _calculate_danger(emotion)
            │       │       → "happy"→GERING, "neutral"/"surprise"→MITTEL,
            │       │          "sad"/"fear"/"disgust"→HOCH, "angry"→EXTREM
            │       └── person_dict = { titel, geschlecht, alter, stimmung, gefahr, beschreibung }
            │
            ├── collected_faces.append(person_dict)
            ├── len(collected_faces) >= expected_face_count?
            └── JA → finalize_and_send_batch()

            Schritt 5: Pool auffüllen
            ├── _append_pool_people(collected_faces)
            │       ├── real_count = 1 (Beispiel) → fehlende_slots = 3
            │       ├── PoolLoader.get_pool_persons(3) → zufällige Pool-Personen
            │       ├── _build_person_dict(..., source="pool") für jede Pool-Person
            │       └── IDs vereinheitlichen: FACE1, FACE2, FACE3, FACE4
            └── data_finalized.emit("BATCH", [4 Personen])
                    → PipelineWorker.result_ready.emit("BATCH", data)
                    → GUI.handle_pipeline_result("BATCH", data)   [QueuedConnection]
```

---

## Phase 6 — GUI empfängt Batch-Signal

```
ScalingAkteGUI.handle_pipeline_result(status, personen_daten)
    │
    ├── status == "EMPTY" → Lade-Spinner ausblenden, close_folder("empty_result")
    │
    └── status == "BATCH"
            ├── _load_person_data_from_final()
            │   ├── DescriptionRepository.read_faces_log_count()
            │   ├── DescriptionRepository.read_deepface_data(faceN)
            │   ├── DescriptionRepository.read_ollama_description(faceN)
            │   │   oder read_moondream_description(faceN)
            │   ├── _build_person_dict_from_final(faceN)
            │   └── _append_pool_people(...)  [GUI-seitig]
            └── handle_new_dataset(personen_daten_aus_final)
                    ├── personen_daten normalisieren (_normalize_person_data)
                    ├── Pipeline-Timeout-Timer stoppen
                    ├── YOLOWorker → PRESENCE_MONITORING vorbereiten
                    │               (_auto_close_monitoring_pending = True)
                    │
                    ├── Mappe schon offen?
                    │   └── JA  → show_flip_video()  (Umblätter-Animation, dann show_open_folder)
                    └── Mappe zu?
                        └── show_animation_with_timer()
```

---

## Phase 7 — Mappe öffnen (Animation)

```
show_animation_with_timer()
    ├── Developer-Modus?
    │   └── JA → start_animation() direkt (kein Countdown)
    └── NEIN → _start_wait_timer()
                    ├── CircularTimerItem anzeigen (Countdown-Kreis)
                    ├── wait_timer startet (33ms Takt = ~30 FPS)
                    └── _update_wait_timer() → Fortschritt aktualisieren
                            └── Countdown = 0 → wait_timer_item.hide()
                                             → start_animation()

start_animation(video_path=open_animation.mp4)
    ├── GUIState → OPENING
    ├── hide_loading_indicator()
    ├── cv2.VideoCapture(open_animation.mp4) öffnen
    └── update_video_frame()  [QTimer-Kette, animation_speed ms/Frame]
            ├── Frame lesen → QPixmap → Szene aktualisieren
            ├── Nächstes Frame in animation_speed ms anfordern
            └── Video fertig → show_open_folder()
```

---

## Phase 8 — Mappe angezeigt

```
show_open_folder()
    ├── GUIState → RESULTS_READY
    ├── _sound_svc.play("folder_open")
    ├── _is_open = True
    ├── Szene aufbauen:
    │       ├── Hintergrundbild (open_folder.png)
    │       ├── setup_ui_elements()
    │       │       └── PersonContainer × 4
    │       │               ├── Bildquelle: sketch/faceN.png oder Pool-Bild
    │       │               ├── Sketch-Darstellung (create_advanced_sketch → Strichzeichnung)
    │       │               ├── Biometrie-Labels (Alter, Geschlecht, Emotion, Gefahrenstufe)
    │       │               └── Beschreibungstext (Typewriter-Animation)
    │       └── setup_buttons()
    │               ├── AnimatedGraphicsButton (Sprache DE/EN)
    │               └── AnimatedGraphicsButton (Reset)
    │
    ├── update_descriptions_from_files()
    │   ├── liest final/faceN_ollama.yaml oder final/faceN_moondream.yaml
    │   └── liest final/faceN_deepface.yaml
    │       → aktualisiert Container nach dem Öffnen nochmals direkt aus final/
    │
    └── Auto-Close aktivieren?
            └── _auto_close_monitoring_pending = True?
                └── JA → GUIState → PRESENCE_MONITORING
                         presence_monitoring_requested.emit()
                         → YOLOWorker.start_presence_monitoring()
                              → state = PRESENCE_MONITORING
```

### 8.1 Typewriter-Animation
```
PersonContainer.trigger_typing()
    └── UiTypewriter.start_typing()
            └── Buchstabe für Buchstabe ausgeben (QTimer)
                mit optionalem Tipp-Sound pro Buchstabe
```

### 8.2 Auto-Close-Überwachung
```
YOLOWorker  [PRESENCE_MONITORING]
    └── PersonPhotoCapture.presence_mode()
            ├── Frame lesen
            ├── YOLO-Inferenz: Person vorhanden?
            └── person_presence_changed.emit(True/False)

GUI: on_person_presence_changed(is_present)
    ├── is_present = True  → _stop_no_person_timer() (Zähler reset)
    └── is_present = False → _missed_presence_checks += 1
            ├── _get_warning_start_missed_checks() erreicht?
            │   └── JA → Reset-Button blinkt (400ms Takt)
            └── _get_auto_close_missed_check_limit() erreicht?
                └── JA → close_folder("auto_close")
```
**Formel:** `limit = close_on_no_person_seconds × 1000 / no_person_check_interval_ms`
Beispiel: 10s Timeout, 2000ms Intervall → 5 verpasste Checks bis Auto-Close.

### 8.3 Reset-Button (Nutzer)
```
btn_reset.clicked → reset_logic()
    └── _start_reset_countdown()
            ├── btn_reset-Bild gegen leere Variante tauschen
            ├── ResetCountdownItem (Zahl) über Reset-Button positionieren
            └── _reset_countdown_timer.start(1000)  [1s Takt]
                    └── _update_reset_countdown()
                            ├── remaining -= 1
                            ├── ResetCountdownItem aktualisieren
                            └── remaining == 0 → close_folder("manual_countdown")
```

### 8.4 Sprache wechseln
```
btn_language.clicked → _on_language_button_clicked()
    ├── Cooldown aktiv? → ignorieren
    └── switch_language_logic()
            ├── btn_language.is_toggled? → "en" : "de"
            ├── _apply_language_to_containers(language)
            ├── TranslationService(target_lang) neu erstellen
            ├── _refresh_descriptions_for_language()
            │       └── Jeden Container-Text neu übersetzen + Typewriter neu starten
            └── _save_language_to_config(language)
        _start_language_button_cooldown()  (5s gesperrt)
```

---

## Phase 9 — Mappe schließen

```
close_folder(reason)
    ├── _clear_pipeline_outputs_on_close setzen
    │       → True für: "manual", "manual_countdown", "auto_close",
    │                   "pipeline_timeout", "empty_result"
    ├── Animation läuft gerade?
    │   └── JA → _pending_close merken (wird nach Animation ausgeführt)
    └── _is_open = True und animated = True?
        └── JA → start_animation(close_animation.mp4, end_callback=show_closed_folder)
```

```
start_animation(close_animation.mp4)
    └── update_video_frame()  [QTimer-Kette]
            └── Video fertig → show_closed_folder()
```

```
show_closed_folder()
    ├── _clear_pipeline_outputs_on_close?
    │   └── JA → _clear_pipeline_output_dirs()
    │               ├── General ordner/final/       leeren
    │               └── General ordner/ollama_inbox/ leeren
    ├── _is_open = False
    ├── GUIState → IDLE
    ├── Alle Timer stoppen (_stop_no_person_timer, wait_timer, ...)
    ├── scene.clear() → active_containers = []
    ├── _sound_svc.play("folder_close")
    ├── Szene aufbauen: geschlossener Ordner + Logos + Kamera-Preview
    └── folder_closed.emit()
            → YOLOWorker.start_capture_mode()
                    └── state = IDLE  → Kamera läuft wieder
```

**PipelineManager (im PipelineWorker-Thread):**
```
Nächster check_for_updates()-Aufruf:
    └── final/ ist leer → keine neuen Dateien → Warten auf neuen faces_log.yaml
        (PipelineManager setzt intern batch state zurück wenn neues faces_log.yaml erscheint)
```

---

## Vollständiger Kreislauf

```
show_closed_folder()
    │
    └── folder_closed.emit() → YOLOWorker.start_capture_mode()
                                        │
                                        ▼
                              ← Phase 2: Kamera-Scan (IDLE) ←
```

---

## Dateien und ihre Rollen

| Datei / Ordner | Rolle |
|---|---|
| `config.yaml` | Zentrale Konfiguration (Sprache, Modelle, Timeouts, Pfade) |
| `path_service.py` | Einzige Stelle für alle Ordnerpfade, `base_dir` steuerbar |
| `General ordner/main_image/face_trigger.jpg` | Kamera-Foto → Eingang für alle KI-Container |
| `General ordner/final/faces_log.yaml` | YOLO-Log: wie viele Gesichter erwartet werden |
| `General ordner/final/face1_deepface.yaml` | Ergebnis von DeepFace (Alter, Geschlecht, Emotion) |
| `General ordner/ollama_ai/ollama_inbox/face1_ollama.yaml` | Moondream-Zwischenergebnis → Eingang für Ollama |
| `General ordner/final/face1_ollama.yaml` | Finales Ollama-Ergebnis (Kriminalgeschichte) |
| `General ordner/final/face1_moondream.yaml` | Fallback-Textausgabe, wenn Ollama deaktiviert ist |
| `General ordner/sketch/face1.png` | Von Face-YOLO erzeugter Gesichts-Crop für die GUI-Anzeige |
| `./pool/` | Vorbefüllte Pool-Personen als Fallback wenn < 4 Gesichter erkannt |
| `translation_cache.yaml` | Lokaler Übersetzungs-Cache (Offline-Betrieb im Museum) |

---

## Threads und Prozesse

| Komponente | Typ | Kommunikation |
|---|---|---|
| Qt-Eventschleife + GUI | Hauptthread | — |
| `YOLOWorker` | QThread | Signale: `frame_ready`, `photo_done`, `person_presence_changed` |
| `PipelineWorker` | QThread | Signal: `result_ready` → `handle_pipeline_result` (QueuedConnection) |
| `ollama_worker.py` | Subprozess (subprocess.Popen) | Dateisystem: liest ollama_inbox/, schreibt final/ |
| Face-YOLO-Container | Docker | Dateisystem: liest main_image/, schreibt sketch/ + deepface_inbox/ + moondream_inbox/ + final/faces_log.yaml |
| Moondream-Container | Docker | Dateisystem: liest moondream_inbox/, schreibt ollama_inbox/ |
| DeepFace-Container | Docker | Dateisystem: liest deepface_inbox/, schreibt final/ |

Alle Signale zwischen Threads werden über Qt `QueuedConnection` gesetzt,
damit GUI-Updates immer im Hauptthread erfolgen.

---

## Ordner-Pfade konfigurieren

Standard: `General ordner/`. Um den gesamten Arbeitsordner umzubenennen
(z.B. beim Zusammenlegen mit docker-compose-Dateien), reicht eine einzige
Änderung in `config.yaml`:

```yaml
paths:
  base_dir: general_ordner
```

Alle Ordner (`final`, `main_image`, `ollama_inbox`, `deepface_inbox`,
`moondream_inbox`, `sketch_dir`, `face_yolo_weights`) folgen automatisch.
Einzelne Pfade können zusätzlich separat überschrieben werden.
