"""
constants.py
------------
Zentrale Konstanten der Anwendung.

ZWECK: Verhindert, dass String-Literale (z.B. KI-Modell-IDs) über
mehrere Dateien verstreut sind. Ein Tippfehler in einer Modell-ID
würde die Pipeline still zum Absturz bringen, ohne Fehlermeldung.
Durch diese Datei gibt es genau EINE Stelle, die geändert werden muss.

VERWENDUNG:
    from constants import PipelineStage, ConfigKey
    entry = config_service.get_pipeline_entry(config, PipelineStage.DEEPFACE)

AUTOREN: Florian Hoeft
"""


class PipelineStage:
    """
    IDs der KI-Modelle in der Pipeline.
    Diese Strings entsprechen exakt den 'id'-Feldern in config.yaml.
    Änderungen hier müssen auch in config.yaml nachgezogen werden.
    """
    MOONDREAM = "moondream"  # Visual Description (VLM) - läuft in Docker
    OLLAMA    = "ollama"     # Kriminalgeschichte - läuft lokal via workers/ollama_worker.py
    DEEPFACE  = "deepface"   # Emotionserkennung - läuft in Docker
    FER       = "fer"        # Emotionserkennung (Fallback) - läuft in Docker


class ConfigKey:
    """
    Häufig verwendete Schlüssel der config.yaml.
    Verhindert Tippfehler bei Dict-Zugriffen auf die Konfiguration.
    """
    PIPELINE    = "pipeline"
    LANGUAGE    = "language"
    ENABLED     = "enabled"
    POOL        = "pool"
    SOUNDS      = "sounds"
    PATHS       = "paths"
    LLM_MODEL   = "llm_model"
    CONFIG_FILE = "config.yaml"
