from deep_translator import GoogleTranslator


class TranslationService:
    """Service zur automatisierten Übersetzung der KI-Ergebnisse."""

    def __init__(self, target_lang='de'):
        # Initialisierung mit automatischer Quellsprachenerkennung
        self.translator = GoogleTranslator(source='auto', target=target_lang)

    def translate_text(self, text):
        """Übersetzt Text oder liefert bei Fehlern das Original zurück (Fallback)."""
        if not text:
            return ""

        try:
            return self.translator.translate(text)
        except Exception as e:
            # Fehlerprotokollierung (z.B. bei Internetproblemen)
            print(f"Translation Error: {e}")
            return text