from deep_translator import GoogleTranslator


class TranslationService:
    """
    Service zur Übersetzung der KI-Ergebnisse.

    Da die lokal genutzten Modelle (Moondream, Ollama) primär englische Texte
    liefern, sorgt dieser Dienst für die Lokalisierung der Fallakten-Inhalte.

    AUTOR: Dennis Penner
    """

    def __init__(self, target_lang='de'):
        # Initialisierung des GoogleTranslators mit automatischer Quellsprachenerkennung.
        # Dies ist die einzige Komponente des Systems, die eine Internetverbindung benötigt.
        self.translator = GoogleTranslator(source='auto', target=target_lang)

    def translate_text(self, text):
        """
        Übersetzt den übergebenen Text in die Zielsprache.

        Sollte die Übersetzung fehlschlagen (z. B. wegen fehlender Internetverbindung),
        greift ein Fallback-Mechanismus, der den englischen Originaltext zurückgibt,
        um den Programmfluss nicht zu unterbrechen.
        """
        if not text:
            return ""

        try:
            return self.translator.translate(text)
        except Exception as e:
            # Protokollierung des Fehlers und Rückgabe des Originaltextes als Sicherheit.
            print(f"Translation Error: {e}")
            return text