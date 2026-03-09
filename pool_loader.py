import copy
import os
import random

import yaml


class PoolLoader:
    def __init__(self, config_path="config.yaml", config_data=None):
        self.config_path = config_path
        self.enabled = False
        self.pool_path = os.path.abspath("./pool")
        self.max_extra_persons = 3
        self.cooldown_batches = 0
        self.pool_persons = []
        self.recent_batches = []
        self.reload(config_data=config_data)

    def reload(self, config_data=None):
        config = config_data if isinstance(config_data, dict) else self._load_config()
        pool_cfg = config.get("pool", {}) if isinstance(config, dict) else {}

        self.enabled = bool(pool_cfg.get("enabled", False))
        self.pool_path = os.path.abspath(pool_cfg.get("path", "./pool"))

        try:
            max_extra = int(pool_cfg.get("max_extra_persons", 3))
        except (TypeError, ValueError):
            max_extra = 3
        self.max_extra_persons = max(0, min(3, max_extra))

        try:
            cooldown_batches = int(pool_cfg.get("cooldown_batches", 0))
        except (TypeError, ValueError):
            cooldown_batches = 0
        self.cooldown_batches = max(0, cooldown_batches)

        self.pool_persons = self._load_pool_persons()
        self._trim_recent_batches()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            return {}

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as exc:
            print(f"[POOL] Konnte Config nicht lesen: {exc}")
            return {}

    def _load_pool_persons(self):
        if not self.enabled:
            return []

        if not os.path.isdir(self.pool_path):
            print(f"[POOL] Pool-Ordner fehlt: {self.pool_path}")
            return []

        persons = []
        for entry in sorted(os.listdir(self.pool_path)):
            person_dir = os.path.join(self.pool_path, entry)
            if not os.path.isdir(person_dir):
                continue

            person_data = self._load_pool_person(person_dir, entry)
            if person_data is not None:
                persons.append(person_data)

        print(f"[POOL] {len(persons)} Pool-Personen geladen.")
        return persons

    def _load_pool_person(self, person_dir, folder_name):
        face_path = os.path.join(person_dir, "face.jpg")
        deepface_path = os.path.join(person_dir, "deepface.yaml")
        moondream_path = os.path.join(person_dir, "moondream.yaml")

        if not os.path.exists(face_path):
            print(f"[POOL] face.jpg fehlt in {person_dir}")
            return None
        if not os.path.exists(deepface_path):
            print(f"[POOL] deepface.yaml fehlt in {person_dir}")
            return None
        if not os.path.exists(moondream_path):
            print(f"[POOL] moondream.yaml fehlt in {person_dir}")
            return None

        try:
            with open(deepface_path, "r", encoding="utf-8") as f:
                deepface_data = yaml.safe_load(f) or {}
            with open(moondream_path, "r", encoding="utf-8") as f:
                moondream_data = yaml.safe_load(f) or {}
        except Exception as exc:
            print(f"[POOL] Fehler beim Laden von {person_dir}: {exc}")
            return None

        return {
            "face_id": f"pool_{folder_name}",
            "face_image_path": face_path,
            "deepface": deepface_data,
            "moondream": moondream_data,
            "source": "pool",
        }

    def get_pool_persons(self, count):
        if not self.enabled:
            return []

        if count <= 0:
            return []

        count = min(int(count), self.max_extra_persons)
        if count <= 0:
            return []

        available = list(self.pool_persons)
        if not available:
            return []

        fallback_to_oldest = False
        available = self._apply_cooldown(available)
        if not available:
            available = self._oldest_cooled_down_persons()
            fallback_to_oldest = True

        if count >= len(available):
            selected = list(available)
            if self.cooldown_batches <= 0:
                random.shuffle(selected)
        else:
            if self.cooldown_batches <= 0:
                selected = random.sample(available, count)
            elif fallback_to_oldest:
                selected = list(available[:count])
            else:
                selected = random.sample(available, count)

        self._remember_batch(selected)
        return [copy.deepcopy(person) for person in selected]

    def _apply_cooldown(self, available):
        if self.cooldown_batches <= 0:
            return list(available)

        blocked_ids = set()
        for batch in self.recent_batches[-self.cooldown_batches:]:
            blocked_ids.update(batch)

        return [person for person in available if person.get("face_id") not in blocked_ids]

    def _remember_batch(self, selected):
        if self.cooldown_batches <= 0:
            return

        batch_ids = [person.get("face_id") for person in selected if person.get("face_id")]
        if not batch_ids:
            return

        self.recent_batches.append(batch_ids)
        self._trim_recent_batches()

    def _trim_recent_batches(self):
        if self.cooldown_batches <= 0:
            self.recent_batches = []
            return

        if len(self.recent_batches) > self.cooldown_batches:
            self.recent_batches = self.recent_batches[-self.cooldown_batches:]

    def _oldest_cooled_down_persons(self):
        if not self.pool_persons:
            return []

        last_seen_index = {}
        for index, batch in enumerate(self.recent_batches):
            for face_id in batch:
                last_seen_index[face_id] = index

        return sorted(
            self.pool_persons,
            key=lambda person: last_seen_index.get(person.get("face_id"), -1)
        )
