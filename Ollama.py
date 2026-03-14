# zur verwendung Ollama installieren. Entweder über die offizielle Webseite (https://ollama.com/) oder über den PS-Befehl: "irm https://ollama.com/install.ps1 | iex"
# nach installation PyCharm neu starten und in das projekt pullen (ollama pull qwen2.5:3b)
# ollama bibliothek installieren: pip install ollama

import ollama
import os
import yaml
import os


def _load_llm_model_from_config(default_model="qwen2.5:3b"):
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        return default_model
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception:
        return default_model
    model = (config or {}).get("llm_model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return default_model


def generate_ai_response(prompt, model=None):
    """
    Sendet einen Prompt an die lokale Ollama-Instanz und gibt die Antwort zurueck.

    Args:
        prompt (str): Die Anweisung oder Frage an die KI.
        model (str | None): Das zu verwendende Modell. Wenn None, wird config.yaml genutzt.

    Returns:
        str: Die generierte Antwort der KI oder eine Fehlermeldung.
    """
    try:
        # Verbindung zu Ollama herstellen und Prompt verarbeiten
        if model is None:
            model = _load_llm_model_from_config()
        print("Current LLM Model:", model)
        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'user', 'content': prompt},
            ],
        )

        # Den Textgehalt der Antwort extrahieren
        return response['message']['content']

    except Exception as e:
        return f"Fehler bei der Kommunikation mit Ollama: {str(e)}"


# --- Beispiel fuer die Nutzung ---
if __name__ == "__main__":
    my_prompt = "Schreibe einen Kriminalbericht ueber eine fiktive Person. die Personenbeschreibung ist bereits erfolgt. Schreibe nur, was die person verbrochen haben koennte in einem Fliesstext. BEachte dabei, dass es sich um ein Verbrechen in der Stasi handelt. Die ausgabe soll nicht mehr als 50 Woerter haben. Antworte moeglichst kurz und versuche etwas lustiges in die story einzubauen. Die story muss nicht erklaert sein. Es reicht einfach nur ein Verbrechen darzustellen"

    print("KI denkt nach...")
    story = generate_ai_response(my_prompt, "llama3.2:1b")

    print("\n--- Generierte Story ---")
    print(story)
