from gui.config_service import ConfigService
from gui.description_repository import DescriptionRepository
from gui.gui_state import GUIState


def main():
    config_service = ConfigService(default_llm_value="test-llm")
    config = config_service.load()
    assert "llm_model" in config

    repo = DescriptionRepository()
    if repo.exists():
        _ = repo.read_moondream_description(0)
        _ = repo.read_deepface_data(0)

    assert GUIState.OPEN.value == "open"
    print("refactor_smoke_test ok")


if __name__ == "__main__":
    main()
