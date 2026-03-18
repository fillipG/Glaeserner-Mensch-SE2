"""
Name: "gui_constants.py"
Beschreibung: Enthält zentrale GUI-Konstanten und Dateipfade.
Autor: Fillip Giffhorn
"""

from path_service import get_paths

SCENE_WIDTH = 1920
SCENE_HEIGHT = 1080
PROJECT_PATHS = get_paths()

PATHS = {
    "closed_folder": "pictures/Akte_V1_Zu.png",
    "open_folder": "pictures/Akte_V3.png",
    "open_animation": "pictures/Akte_animation.mov",
    "close_animation": "pictures/Akte_animation_reverse.mov",
    "flip_animation": "pictures/Akte_umblaettern.mov",
    "reset_button": "pictures/reset_button.png",
    "reset_button_empty": "pictures/reset_button_empty.png",
    "language_de": "pictures/change_language_german.png",
    "language_en": "pictures/change_language_english.png",
    "logo_bmftr": "pictures/Logo 1 - BMFTR_de_Web_RGB_gef_durch.jpg",
    "logo_ki_owl": "pictures/Logo 2 - KI_Akademie_OWL_Logo_guer_rgb.png",
    "logo_th_owl": "pictures/TH_OWL_Logo.png",
    # GUI-Code arbeitet an mehreren Stellen mit APIs, die historisch String-Pfade
    # erwarten. Deshalb werden die zentralen Path-Objekte hier bewusst in Strings
    # ueberfuehrt, waehrend path_service.py weiterhin Path zurueckgibt.
    "sketch_dir": str(PROJECT_PATHS["sketch_dir"]),
    "final_dir": str(PROJECT_PATHS["final"]),
    "ollama_inbox": str(PROJECT_PATHS["ollama_inbox"]),
}
