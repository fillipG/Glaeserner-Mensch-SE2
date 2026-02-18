from deep_translator import GoogleTranslator

class TranslationService:
    def __init__(self, target_lang='de'):
        self.translator = GoogleTranslator(source='auto', target=target_lang)

    def translate_text(self, text):
        if not text: return ""
        try:
            return self.translator.translate(text)
        except Exception as e:
            print(f"🌐 Translation Error: {e}")
            return text # Fallback auf Original