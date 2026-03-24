"""
SoundService
------------
Zentraler Service fuer alle App-Sounds.
Kapselt Laden, Abspielen, Lautstaerke und enabled/disabled.
"""

from pathlib import Path
import time

from PyQt6.QtCore import QUrl
from PyQt6.QtMultimedia import QSoundEffect


class SoundService:
    """Zentraler Service fuer die Wiedergabe aller GUI-Sounds."""

    KNOWN_SOUNDS = ["typewriter", "folder_open", "folder_close"]
    TYPEWRITER_MIN_INTERVAL_S = 0.08

    def __init__(self, config: dict):
        sounds_cfg = config.get("sounds", {})
        if not isinstance(sounds_cfg, dict):
            sounds_cfg = {}

        self._enabled = bool(sounds_cfg.get("enabled", True))
        self._volume = max(0.0, min(1.0, float(sounds_cfg.get("volume", 0.3))))
        self._effects: dict[str, QSoundEffect] = {}
        self._last_typewriter_play = 0.0

        repo_root = Path(__file__).resolve().parents[1]
        for sound_name in self.KNOWN_SOUNDS:
            rel_path = sounds_cfg.get(sound_name, "")
            if not rel_path:
                continue
            abs_path = repo_root / rel_path
            if not abs_path.exists():
                print(f"[SOUND] Datei nicht gefunden: {abs_path}")
                continue
            effect = QSoundEffect()
            effect.setSource(QUrl.fromLocalFile(str(abs_path)))
            effect.setVolume(self._volume)
            effect.setLoopCount(1)
            self._effects[sound_name] = effect
            print(f"[SOUND] Geladen: {sound_name}")

    def play(self, sound_name: str, char: str = "") -> None:
        """Spielt einen bekannten Sound ab und filtert Leerzeichen bei Typewriter-Sound."""
        if not self._enabled:
            return
        if sound_name == "typewriter":
            if not char or char.isspace():
                return
            now = time.monotonic()
            if (now - self._last_typewriter_play) < self.TYPEWRITER_MIN_INTERVAL_S:
                return
            self._last_typewriter_play = now
        effect = self._effects.get(sound_name)
        if effect is None:
            return
        effect.play()

    def set_enabled(self, enabled: bool) -> None:
        """Aktualisiert den globalen Enabled-Zustand fuer Sounds."""
        self._enabled = bool(enabled)

    def set_volume(self, volume: float) -> None:
        """Setzt die globale Lautstaerke fuer alle geladenen Sounds."""
        self._volume = max(0.0, min(1.0, float(volume)))
        for effect in self._effects.values():
            effect.setVolume(self._volume)

    @property
    def enabled(self) -> bool:
        """Aktueller Enabled-Zustand."""
        return self._enabled
