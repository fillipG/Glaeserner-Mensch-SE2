# zur verwendung Ollama installieren. Entweder über die offizielle Webseite (https://ollama.com/) oder über den PS-Befehl: "irm https://ollama.com/install.ps1 | iex"
# nach installation PyCharm neu starten und in das projekt pullen (ollama pull llama3.2:1b)
# ollama bibliothek installieren: pip install ollama

import ollama


def generate_ai_response(prompt, model="llama3.2:1b"):
    """
    Sendet einen Prompt an die lokale Ollama-Instanz und gibt die Antwort zurück.

    Args:
        prompt (str): Die Anweisung oder Frage an die KI.
        model (str): Das zu verwendende Modell (z.B. 'llama3', 'mistral', 'llama4').

    Returns:
        str: Die generierte Antwort der KI oder eine Fehlermeldung.
    """
    try:
        # Verbindung zu Ollama herstellen und Prompt verarbeiten
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


# --- Beispiel für die Nutzung ---
if __name__ == "__main__":
    my_prompt = "Schreibe einen Kriminalbericht über eine fiktive Person. die Personenbeschreibung ist bereits erfolgt. Schreibe nur, was die person verbrochen haben könnte in einem Fließtext. BEachte dabei, dass es sich um ein Verbrechen in der Stasi handelt. Die ausgabe soll nicht mehr als 50 Wörter haben. Antworte möglichst kurz und versuche etwas lustiges in die story einzubauen. Die story muss nicht erklärt sein. Es reicht einfach nur ein Verbrechen darzustellen"

    print("KI denkt nach...")
    story = generate_ai_response(my_prompt, model="llama3.2:1b")

    print("\n--- Generierte Story ---")
    print(story)