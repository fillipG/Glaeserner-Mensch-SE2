"""
gui/mixins/
-----------
Mixin-Klassen für die ScalingAkteGUI.

Jede Mixin-Datei enthält thematisch zusammengehörige Methoden der
Haupt-GUI-Klasse. Alle Mixins teilen sich 'self' mit ScalingAkteGUI
und können daher auf alle Instanzvariablen zugreifen.

Übersicht:
    animation_mixin.py      – Ordner-Videos, Öffnungs-/Schließ-Animationen
    camera_mixin.py         – Kamera-Preview, Gesichtserkennung im Vorschaubild
    presence_mixin.py       – Auto-Close, Anwesenheits-Timer, Warnungen
    language_mixin.py       – Sprachwechsel, Button-Cooldown, Container-Update
    reset_mixin.py          – Reset-Countdown, Button-Zustand
    config_handlers_mixin.py– Admin-Menü-Signale, Config lesen/schreiben
"""
