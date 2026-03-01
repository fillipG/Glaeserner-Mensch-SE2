import ollama


def generate_ai_response(prompt, model="llama3"):
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
    my_prompt = "Schreibe einen Kriminalbericht über eine fiktive Person. Die ausgabe soll nicht mehr als 300 Zeichen haben"

    print("KI denkt nach...")
    story = generate_ai_response(my_prompt, model="llama3")  # Ändere 'llama3' zu deinem installierten Modell

    print("\n--- Generierte Story ---")
    print(story)