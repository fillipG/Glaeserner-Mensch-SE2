import os
from typing import List


class ImageDeleter:
    # Konstruktor mit optionalem Ordnerpfad
    def __init__(self, folder_path: str = None):
        self.folder_path = folder_path

    # Setzt oder ändert den Ordnerpfad
    def set_folder_path(self, folder_path: str) -> None:
        self.folder_path = folder_path

    # Gibt alle Bilddateien im Ordner zurück
    def get_images(self) -> List[str]:
        if not self.folder_path:
            raise ValueError("Folder path is not set.")

        if not os.path.isdir(self.folder_path):
            raise FileNotFoundError("The specified folder does not exist.")

        image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")

        return [
            file for file in os.listdir(self.folder_path)
            if file.lower().endswith(image_extensions)
        ]

    # Löscht alle Bilder im Ordner
    def delete_all_images(self) -> None:
        images = self.get_images()

        for image in images:
            image_path = os.path.join(self.folder_path, image)
            os.remove(image_path)

    # Löscht ein bestimmtes Bild
    def delete_image(self, image_name: str) -> None:
        if not self.folder_path:
            raise ValueError("Folder path is not set.")

        image_path = os.path.join(self.folder_path, image_name)

        if not os.path.isfile(image_path):
            raise FileNotFoundError("The specified image does not exist.")

        os.remove(image_path)
