"""
service.py
----------
Übersetzungs-Service für KI-generierte Texte.

Da die lokalen Modelle (Moondream, Ollama) primär englische Texte liefern,
sorgt dieser Service für die Lokalisierung der Fallakten-Inhalte.

WICHTIG FÜR MUSEUM-BETRIEB:
    Die Google Translate API benötigt eine Internetverbindung.
    Falls die Verbindung ausfällt (häufig in Ausstellungsumgebungen),
    greift ein lokaler Datei-Cache (translation_cache.yaml).
    Einmal übersetzte Texte werden aus dem Cache geladen, ohne Internetzugang.
    Bei komplettem Cache-Miss und fehlender Verbindung: englischer Originaltext.

AUTOR: Dennis Penner
"""

import os
import yaml
from deep_translator import GoogleTranslator


# Lokale Cache-Datei für Übersetzungen (verhindert wiederholte API-Aufrufe)
CACHE_FILE = "translation_cache.yaml"


class TranslationService:
    """
    Übersetzt Texte via Google Translate mit lokalem Datei-Cache.

    Der Cache speichert alle Übersetzungen dauerhaft in translation_cache.yaml.
    Damit funktioniert die Anzeige auch bei vorübergehendem Internet-Ausfall,
    solange die Texte bereits mindestens einmal übersetzt wurden.
    """

    def __init__(self, target_lang='de'):
        """
        :param target_lang: Zielsprache (z.B. 'de' für Deutsch, 'en' für Englisch).
        """
        self._target_lang = target_lang
        # GoogleTranslator erkennt die Quellsprache automatisch
        self.translator = GoogleTranslator(source='auto', target=target_lang)
        # Cache beim Start laden (auch wenn leer oder nicht vorhanden)
        self._cache = self._load_cache()

    def _load_cache(self):
        """
        Lädt den lokalen Übersetzungs-Cache aus translation_cache.yaml.
        Bei fehlender Datei oder Fehler: leeres Dict zurückgeben.
        """
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
        except Exception:
            pass
        return {}

    def _save_cache(self):
        """
        Speichert den aktuellen Cache in translation_cache.yaml.
        Fehler beim Speichern werden ignoriert damit der Hauptfluss nicht stoppt.
        """
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                yaml.safe_dump(self._cache, f, allow_unicode=True)
        except Exception:
            pass  # Cache-Fehler darf die Anwendung nicht unterbrechen

    def translate_text(self, text):
        """
        Übersetzt den Text in die Zielsprache.

        Reihenfolge:
        1. Cache prüfen → sofort zurückgeben ohne Netzwerk
        2. Google Translate API aufrufen
        3. Ergebnis im Cache speichern für nächste Verwendung
        4. Bei Fehler: englischen Originaltext zurückgeben

        :param text: Zu übersetzender Text.
        :return: Übersetzter Text oder Original bei Fehler.
        """
        if not text:
            return ""

        # Cache-Key enthält die Zielsprache, damit DE- und EN-Übersetzungen
        # nicht kollidieren wenn die Sprache gewechselt wird.
        cache_key = f"{self._target_lang}:{text}"

        # FALL 1: Übersetzung bereits im Cache → kein Netzwerkzugriff nötig
        if cache_key in self._cache:
            return self._cache[cache_key]

        # FALL 2: Nicht im Cache → Google Translate API aufrufen
        try:
            translated = self.translator.translate(text)
            # Erfolgreiche Übersetzung im Cache speichern
            self._cache[cache_key] = translated
            self._save_cache()
            return translated
        except Exception as e:
            # Netzwerkfehler oder API-Fehler → englischen Text als Fallback
            print(f"Translation Error (kein Internet?): {e}")
            return text
