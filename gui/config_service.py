"""
gui/config_service.py
---------------------
Kompatibilitäts-Shim.

Der ConfigService wurde ins Root-Verzeichnis verschoben (config_service.py),
da er nicht nur von der GUI, sondern auch von main.py, PipelineManager und
LocalWorkerManager verwendet wird.

Dieser Shim stellt sicher, dass bestehende Imports innerhalb von gui/
(z.B. 'from .config_service import ConfigService') weiterhin funktionieren.
"""

# Re-Export aus dem zentralen Root-Modul
from config_service import ConfigService  # noqa: F401
