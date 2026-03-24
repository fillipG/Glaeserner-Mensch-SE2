# Sound-Dateien

Audio-Feedback fuer die Museums-App. WAV-Dateien werden hier manuell abgelegt.

## Benoetigte Dateien

| Datei | Beschreibung | Dauer |
|-------|-------------|-------|
| typewriter_key.wav | Einzelner Schreibmaschinen-Tastenanschlag | ~50-100ms |
| folder_open.wav | Kurzes Papier-Rascheln (Akte aufklappen) | ~100-200ms |
| folder_close.wav | Kurzes Papier-Rascheln (Akte zuklappen) | ~100-200ms |

## Anforderungen

- Format: WAV, Mono, 16-bit, 44100 Hz
- Lizenz: CC0 (Public Domain)
- Lautstaerke: Dezent, Hintergrund-Atmosphaere

## Empfohlene Quellen (freesound.org, CC0-Filter setzen)

- typewriter: freesound.org/people/yottasounds/sounds/380138/
- folder: Suche "paper rustle short" mit CC0-Filter

## Hinweis

Fehlende Dateien verursachen keinen Crash. Der SoundService loggt
eine Warnung und deaktiviert den jeweiligen Sound automatisch.
