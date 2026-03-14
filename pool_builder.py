#!/usr/bin/env python3
"""
pool_builder.py
===============
Legt neue Pool-Personen unter pool/personNN/ an.

Mit Bildern:
- schneidet das Gesicht analog zum Face-YOLO-Worker aus
- speichert den Crop als face.jpg
- erzeugt deepface.yaml per lokaler DeepFace-Analyse
- erzeugt ollama.yaml per lokalem Ollama-Aufruf
- erlaubt danach eine interaktive Nachbearbeitung der generierten Werte

Ohne Bilder:
- faellt auf den bisherigen Zufallsmodus zur Erzeugung von Pool-Daten zurueck

Verwendung:
  python pool_builder.py
  python pool_builder.py --count 5
  python pool_builder.py --image foto.jpg
  python pool_builder.py --images foto1.jpg foto2.jpg
  python pool_builder.py --image-dir .\meine_bilder
  python pool_builder.py --image-dir .\meine_bilder --no-review
"""

import argparse
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import cv2
import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO

try:
    import ollama
except ImportError:
    ollama = None

try:
    import keras
except ImportError:
    keras = None

try:
    import torch
except ImportError:
    torch = None

try:
    from rembg import new_session, remove
except ImportError:
    new_session = None
    remove = None


BASE_DIR = Path(__file__).resolve().parent
POOL_DIR = BASE_DIR / "pool"
CONFIG_PATH = BASE_DIR / "config.yaml"
FACE_YOLO_WEIGHTS = BASE_DIR / "General ordner" / "docker-compose-face-Yolo" / "yolov8n-face.pt"
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

DEFAULT_OLLAMA_PROMPT = (
    "Write a criminal report about a fictional person.\n"
    "The person has already been described. Write only what crime\n"
    "the person might have committed in a short flowing paragraph.\n"
    "Make sure it is a crime within the Stasi context.\n"
    "The output must be between 30 and 50 words long.\n"
    "Stay within this range and try to make it a little funny.\n"
    "The story does not need to be explained. It is enough to show one crime."
)
POOL_SOURCE_PROMPT = "Auto-generated from face crop and DeepFace attributes by pool_builder."
DEFAULT_FACE_CONFIDENCE = 0.5
DEFAULT_FACE_PADDING = 18
DEFAULT_FACE_PADDING_RATIO = 0.06
HAAR_CASCADE_FILES = (
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_frontalface_default.xml",
)

EMOTIONEN = ["happy", "sad", "angry", "surprised", "fearful", "disgusted", "neutral"]
GESCHLECHTER = ["Mann", "Frau"]
ALTER_RANGE = (7, 95)

DESCRIPTION_TEMPLATES = [
    "Person wears a bright yellow hazmat suit and carries a clipboard. Last seen arguing with a vending machine at the central station.",
    "Person is dressed in a three-piece suit made entirely of aluminum foil. Believed to be a retired cheese inspector from Dusseldorf.",
    "Person wears mismatched socks, one green one red, and a cape made from a shower curtain. Occupation: professional cloud watcher.",
    "Person appears to be wearing pajamas with tiny dinosaurs on them. Claims to be a certified time traveler from the year 2087.",
    "Person sports a leather jacket covered in stickers of medieval knights. Last known job: competitive noodle taster.",
    "Person wears a lab coat with suspicious stains and swimming goggles pushed up on their forehead. Expertise: unknown.",
    "Person is dressed as a viking but carrying a briefcase. The briefcase contains, according to witnesses, only a single banana.",
    "Person wears an oversized trench coat and a name tag that says 'Definitely Not A Robot'. Threat level: unclear.",
    "Person is dressed in all beige. Every item is beige. The expression is also beige. Last seen near a beige building.",
    "Person wears a tuxedo t-shirt and sandals with socks. Believed to be a freelance philosopher specializing in elevator music.",
    "Person has a scarf wrapped seven times around their head despite it being summer. Profession: mystery.",
    "Person wears a sequined jumpsuit and roller skates. Employment status: between heists.",
    "Person appears to be wearing a graduation gown as everyday clothing. Diploma subject: Advanced Procrastination.",
    "Person sports a fanny pack worn backwards and sunglasses indoors. Last verified sighting: snack aisle, unknown supermarket.",
    "Person wears a full knight armor but with running shoes. Currently wanted for questioning regarding a missing garden gnome.",
]

CRIME_STORIES = [
    "Suspected of replacing all the salt shakers in the city with sugar. Motive: unknown. Damage: incalculable.",
    "Last seen leaving a library with 47 overdue books dating back to 1987. Still at large.",
    "Wanted in three districts for rearranging supermarket products into alphabetical order without permission.",
    "Believed to have stolen exactly one sock from 200 different households over the past year.",
    "Under investigation for teaching pigeons to tap-dance outside council meetings.",
    "Suspected of leaving anonymous motivational sticky notes on strangers' cars. Victims describe feeling 'uncomfortably encouraged'.",
    "Wanted for questioning after every clock in a local hotel was set forward by exactly 7 minutes.",
    "Known accomplice in the great pretzel heist of last Tuesday. Crumbs were found at the scene.",
    "Allegedly responsible for signing up an entire office building for a cheese-of-the-month club without consent.",
    "Suspected of installing tiny doors at the base of walls in public buildings. Purpose: undetermined.",
]

_DEEPFACE_MODULE = None


def load_config():
    if not CONFIG_PATH.exists():
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception as exc:
        print(f"[!] Konnte config.yaml nicht lesen: {exc}")
        return {}


def get_face_yolo_confidence(config):
    face_yolo_cfg = config.get("face_yolo", {}) if isinstance(config, dict) else {}
    if isinstance(face_yolo_cfg, dict):
        raw_value = face_yolo_cfg.get("confidence", DEFAULT_FACE_CONFIDENCE)
    else:
        raw_value = DEFAULT_FACE_CONFIDENCE
    try:
        return max(0.10, min(0.90, float(raw_value)))
    except (TypeError, ValueError):
        return DEFAULT_FACE_CONFIDENCE


def get_deepface_settings(config):
    pipeline = config.get("pipeline", []) if isinstance(config, dict) else []
    for model_cfg in pipeline:
        if isinstance(model_cfg, dict) and model_cfg.get("id") == "deepface":
            return {
                "enabled": bool(model_cfg.get("enabled", True)),
                "use_retinaface": bool(model_cfg.get("use_retinaface", True)),
            }
    return {
        "enabled": True,
        "use_retinaface": True,
    }


def get_ollama_settings(config):
    pipeline = config.get("pipeline", []) if isinstance(config, dict) else []
    prompt = DEFAULT_OLLAMA_PROMPT
    enabled = True
    for model_cfg in pipeline:
        if isinstance(model_cfg, dict) and model_cfg.get("id") == "ollama":
            enabled = bool(model_cfg.get("enabled", True))
            prompt = str(model_cfg.get("prompt", DEFAULT_OLLAMA_PROMPT)).strip() or DEFAULT_OLLAMA_PROMPT
            break
    model_name = config.get("llm_model", "qwen2.5:3b")
    if not isinstance(model_name, str) or not model_name.strip():
        model_name = "qwen2.5:3b"
    return {
        "enabled": enabled,
        "prompt": prompt,
        "model": model_name.strip(),
    }


def build_text_sequences(total_count):
    descriptions = list(DESCRIPTION_TEMPLATES)
    crime_stories = list(CRIME_STORIES)
    random.shuffle(descriptions)
    random.shuffle(crime_stories)

    description_sequence = [descriptions[index % len(descriptions)] for index in range(total_count)]
    crime_sequence = [crime_stories[index % len(crime_stories)] for index in range(total_count)]
    return description_sequence, crime_sequence


def get_next_person_index():
    if not POOL_DIR.exists():
        POOL_DIR.mkdir(parents=True, exist_ok=True)
        return 1

    existing = [
        entry.name for entry in POOL_DIR.iterdir()
        if entry.is_dir() and entry.name.startswith("person")
    ]
    numbers = []
    for name in existing:
        try:
            numbers.append(int(name.replace("person", "")))
        except ValueError:
            continue
    return max(numbers) + 1 if numbers else 1


def resolve_image_path(image_path):
    if not image_path:
        return None
    candidate = Path(image_path)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    return candidate if candidate.exists() else None


def gather_images(image_paths=None, image_dir=None):
    resolved_images = []
    seen = set()

    for image_path in image_paths or []:
        resolved = resolve_image_path(image_path)
        if resolved is None:
            print(f"[!] Bild nicht gefunden: {image_path}")
            continue
        if resolved not in seen:
            resolved_images.append(resolved)
            seen.add(resolved)

    if image_dir:
        resolved_dir = resolve_image_path(image_dir)
        if resolved_dir is None or not resolved_dir.is_dir():
            print(f"[!] Bildordner nicht gefunden: {image_dir}")
        else:
            for entry in sorted(resolved_dir.iterdir()):
                if not entry.is_file():
                    continue
                if entry.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                    continue
                absolute_candidate = entry.resolve()
                if absolute_candidate not in seen:
                    resolved_images.append(absolute_candidate)
                    seen.add(absolute_candidate)

    return resolved_images


def write_yaml(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)


def load_image_bgr(image_path):
    image_rgb = np.array(Image.open(image_path).convert("RGB"))
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def preprocess_for_detection(image):
    brightened = cv2.convertScaleAbs(image, alpha=1.08, beta=8)
    lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(enhanced, -1, sharpen_kernel)


def get_yolo_device():
    if torch is None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_face_yolo_model(device):
    if not FACE_YOLO_WEIGHTS.exists():
        raise RuntimeError(f"Face-YOLO Gewicht nicht gefunden: {FACE_YOLO_WEIGHTS}")
    if importlib.util.find_spec("omegaconf") is None:
        raise RuntimeError("Python-Paket 'omegaconf' fehlt fuer das lokale Face-YOLO-Gewicht.")
    model = YOLO(str(FACE_YOLO_WEIGHTS))
    if device == "cuda":
        model.to("cuda")
    return model


def load_haar_face_detector():
    detectors = []
    for cascade_name in HAAR_CASCADE_FILES:
        cascade_path = Path(cv2.data.haarcascades) / cascade_name
        detector = cv2.CascadeClassifier(str(cascade_path))
        if detector.empty():
            continue
        detectors.append((cascade_name, detector))

    if not detectors:
        raise RuntimeError("Keine Haar-Cascade konnte geladen werden.")
    return detectors


def create_rembg_session():
    if new_session is None or remove is None:
        return None
    try:
        return new_session("u2net")
    except Exception as exc:
        print(f"[!] rembg konnte nicht initialisiert werden, nutze Crop ohne Hintergrundentfernung: {exc}")
        return None


def choose_primary_box(results):
    best_box = None
    best_conf = -1.0
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.conf is None:
            continue
        for index, confidence in enumerate(boxes.conf.cpu().tolist()):
            if float(confidence) > best_conf:
                best_conf = float(confidence)
                best_box = boxes.xyxy[index].cpu().numpy()
    return best_box


def choose_largest_face_box(faces):
    if not len(faces):
        return None
    return max(faces, key=lambda item: int(item[2]) * int(item[3]))


def get_face_padding(box_width, box_height, fallback_padding):
    dynamic_padding = int(max(box_width, box_height) * DEFAULT_FACE_PADDING_RATIO)
    return max(10, min(fallback_padding, dynamic_padding if dynamic_padding > 0 else fallback_padding))


def create_face_crop(image_path, yolo_model, device, confidence, padding, rembg_session):
    if yolo_model is None:
        raise RuntimeError("Kein Gesichtsdetektor initialisiert.")

    image = load_image_bgr(image_path)

    detection_image = preprocess_for_detection(image)
    results = yolo_model(detection_image, conf=confidence, device=device, verbose=False)
    best_box = choose_primary_box(results)
    if best_box is None:
        raise RuntimeError("Kein Gesicht gefunden.")

    x1, y1, x2, y2 = [int(value) for value in best_box]
    box_padding = get_face_padding(x2 - x1, y2 - y1, padding)
    height, width = image.shape[:2]
    x1 = max(0, x1 - box_padding)
    y1 = max(0, y1 - box_padding)
    x2 = min(width, x2 + box_padding)
    y2 = min(height, y2 + box_padding)

    face = image[y1:y2, x1:x2]
    if face.size == 0:
        raise RuntimeError("Gesichtscrop ist leer.")

    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)

    if rembg_session is not None and remove is not None:
        face_pil = remove(face_pil, session=rembg_session)

    if face_pil.mode == "RGBA":
        background = Image.new("RGB", face_pil.size, (255, 255, 255))
        background.paste(face_pil, mask=face_pil.getchannel("A"))
        return background
    return face_pil.convert("RGB")


def create_face_crop_with_haar(image_path, detectors, padding, rembg_session):
    image = load_image_bgr(image_path)
    detection_image = preprocess_for_detection(image)
    grayscale_variants = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(detection_image, cv2.COLOR_BGR2GRAY),
    )

    best_face = None
    for _cascade_name, detector in detectors:
        for gray in grayscale_variants:
            faces = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(50, 50))
            candidate = choose_largest_face_box(faces)
            if candidate is None:
                continue
            if best_face is None or int(candidate[2]) * int(candidate[3]) > int(best_face[2]) * int(best_face[3]):
                best_face = candidate
        if best_face is not None:
            break

    if best_face is None:
        raise RuntimeError("Kein Gesicht gefunden.")

    x, y, w, h = [int(value) for value in best_face]
    box_padding = get_face_padding(w, h, padding)
    height, width = image.shape[:2]
    x1 = max(0, int(x) - box_padding)
    y1 = max(0, int(y) - box_padding)
    x2 = min(width, int(x + w) + box_padding)
    y2 = min(height, int(y + h) + box_padding)

    face = image[y1:y2, x1:x2]
    if face.size == 0:
        raise RuntimeError("Gesichtscrop ist leer.")

    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)

    if rembg_session is not None and remove is not None:
        face_pil = remove(face_pil, session=rembg_session)

    if face_pil.mode == "RGBA":
        background = Image.new("RGB", face_pil.size, (255, 255, 255))
        background.paste(face_pil, mask=face_pil.getchannel("A"))
        return background
    return face_pil.convert("RGB")


def _round_or_none(value):
    try:
        return round(float(value), 4)
    except Exception:
        return None


def _confidence_for_label(score_dict, label):
    if not isinstance(score_dict, dict) or not label:
        return None
    return _round_or_none(score_dict.get(label))


def _map_gender_to_de(dominant_gender):
    if dominant_gender == "Man":
        return "Mann"
    if dominant_gender == "Woman":
        return "Frau"
    return dominant_gender


def analyze_with_deepface(face_path, use_retinaface):
    DeepFace = get_deepface_module()
    detector_backend = "retinaface" if use_retinaface else "skip"
    backends_to_try = [detector_backend]
    if detector_backend != "skip":
        backends_to_try.append("skip")

    last_error = None
    for backend in backends_to_try:
        try:
            results = DeepFace.analyze(
                img_path=str(face_path),
                actions=["age", "gender", "emotion"],
                enforce_detection=False,
                detector_backend=backend,
                silent=True,
            )
            res = results[0] if isinstance(results, list) else results
            dominant_emotion = res.get("dominant_emotion")
            dominant_gender = res.get("dominant_gender")
            emotion_confidence = _confidence_for_label(res.get("emotion"), dominant_emotion)
            gender_confidence = _confidence_for_label(res.get("gender"), dominant_gender)
            confidence = emotion_confidence if emotion_confidence is not None else gender_confidence
            return {
                "Emotion": dominant_emotion or "neutral",
                "Alter": int(res.get("age", 0)) if res.get("age") is not None else 0,
                "Geschlecht": _map_gender_to_de(dominant_gender) or "Unbekannt",
                "Emotion_Confidence": emotion_confidence,
                "Gender_Confidence": gender_confidence,
                "Confidence": confidence,
            }
        except Exception as exc:
            last_error = exc
            if backend != backends_to_try[-1]:
                print(f"  [!] DeepFace mit Backend '{backend}' fehlgeschlagen, versuche Fallback 'skip'.")

    raise RuntimeError(f"DeepFace-Analyse fehlgeschlagen: {last_error}")


def get_deepface_module():
    global _DEEPFACE_MODULE
    if _DEEPFACE_MODULE is not None:
        return _DEEPFACE_MODULE

    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    if "tf_keras" not in sys.modules:
        if keras is None:
            raise RuntimeError("Keras ist lokal nicht installiert.")
        sys.modules["tf_keras"] = keras

    try:
        from deepface import DeepFace as deepface_module
    except Exception as exc:
        raise RuntimeError(f"DeepFace konnte nicht importiert werden: {exc}") from exc

    _DEEPFACE_MODULE = deepface_module
    return _DEEPFACE_MODULE


def build_source_description(image_path, deepface_data):
    label = image_path.stem.replace("_", " ").replace("-", " ").strip() or "pool person"
    age = deepface_data.get("Alter", "unknown")
    gender = deepface_data.get("Geschlecht", "unknown")
    emotion = deepface_data.get("Emotion", "neutral")
    return (
        f"Reference label: {label}. "
        f"Approximate age: {age}. "
        f"Perceived gender: {gender}. "
        f"Dominant emotion: {emotion}. "
        "The text was auto-generated from a cropped portrait image for the museum pool."
    )


def windows_creation_flags():
    if os.name != "nt":
        return 0
    return subprocess.CREATE_NO_WINDOW


def is_ollama_api_ready():
    try:
        with urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as response:
            return response.status == 200
    except (URLError, OSError, ValueError):
        return False


def wait_for_ollama_api(timeout_seconds):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if is_ollama_api_ready():
            return True
        time.sleep(0.5)
    return False


def load_available_ollama_models():
    with urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as response:
        payload = json.loads(response.read().decode("utf-8"))
    models = payload.get("models", [])
    names = set()
    for model in models:
        if not isinstance(model, dict):
            continue
        name = model.get("name")
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    return names


def ensure_ollama_ready(model_name):
    if importlib.util.find_spec("ollama") is None or ollama is None:
        raise RuntimeError("Das Python-Paket 'ollama' ist lokal nicht installiert.")

    started_process = None
    if not is_ollama_api_ready():
        ollama_executable = shutil.which("ollama")
        if not ollama_executable:
            raise RuntimeError("Ollama wurde nicht im PATH gefunden.")
        started_process = subprocess.Popen(
            [ollama_executable, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=windows_creation_flags(),
        )
        if not wait_for_ollama_api(timeout_seconds=30):
            raise RuntimeError("Ollama konnte nicht gestartet werden.")

    available_models = load_available_ollama_models()
    if model_name not in available_models:
        raise RuntimeError(f"Das Modell '{model_name}' ist lokal nicht vorhanden.")

    return started_process


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def generate_ollama_description(source_description, ollama_settings):
    prompt_template = ollama_settings.get("prompt", DEFAULT_OLLAMA_PROMPT)
    model_name = ollama_settings.get("model", "qwen2.5:3b")

    if not ollama_settings.get("enabled", True):
        return {
            "prompt": prompt_template,
            "source_prompt": POOL_SOURCE_PROMPT,
            "source_description": source_description,
            "description": source_description,
        }

    prompt = f"Person description: {source_description}\n\n{prompt_template}"
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    description = response["message"]["content"].strip()
    return {
        "prompt": prompt_template,
        "source_prompt": POOL_SOURCE_PROMPT,
        "source_description": source_description,
        "description": description,
    }


def build_random_deepface_yaml():
    emotion = random.choice(EMOTIONEN)
    geschlecht = random.choice(GESCHLECHTER)
    alter = random.randint(*ALTER_RANGE)
    confidence = round(random.uniform(0.55, 0.97), 2)
    gender_confidence = round(random.uniform(0.55, 0.97), 2)
    return {
        "Emotion": emotion,
        "Alter": alter,
        "Geschlecht": geschlecht,
        "Emotion_Confidence": confidence,
        "Gender_Confidence": gender_confidence,
        "Confidence": confidence,
    }


def build_random_ollama_yaml(description, crime_story):
    return {
        "prompt": DEFAULT_OLLAMA_PROMPT,
        "source_prompt": POOL_SOURCE_PROMPT,
        "source_description": description,
        "description": f"{description} {crime_story}",
    }


def prompt_override(label, current_value, cast=None):
    raw_value = input(f"      {label} [{current_value}]: ").strip()
    if not raw_value:
        return current_value
    if cast is None:
        return raw_value
    try:
        return cast(raw_value)
    except ValueError:
        print(f"      [!] Ungueltiger Wert fuer {label}, alter Wert bleibt erhalten.")
        return current_value


def review_generated_data(person_index, deepface_data, ollama_data):
    print(f"\n  [?] Werte fuer person{person_index} anpassen? [j/N]")
    answer = input("      > ").strip().lower()
    if answer not in {"j", "ja", "y", "yes"}:
        return deepface_data, ollama_data

    deepface_data = dict(deepface_data)
    ollama_data = dict(ollama_data)

    deepface_data["Emotion"] = prompt_override("Emotion", deepface_data.get("Emotion", "neutral"))
    deepface_data["Alter"] = prompt_override("Alter", deepface_data.get("Alter", 0), int)
    deepface_data["Geschlecht"] = prompt_override("Geschlecht", deepface_data.get("Geschlecht", "Unbekannt"))
    deepface_data["Emotion_Confidence"] = prompt_override(
        "Emotion_Confidence",
        deepface_data.get("Emotion_Confidence"),
        float,
    )
    deepface_data["Gender_Confidence"] = prompt_override(
        "Gender_Confidence",
        deepface_data.get("Gender_Confidence"),
        float,
    )
    deepface_data["Confidence"] = (
        deepface_data.get("Emotion_Confidence")
        if deepface_data.get("Emotion_Confidence") is not None
        else deepface_data.get("Gender_Confidence")
    )

    ollama_data["source_description"] = prompt_override(
        "source_description",
        ollama_data.get("source_description", ""),
    )
    ollama_data["description"] = prompt_override(
        "description",
        ollama_data.get("description", ""),
    )
    return deepface_data, ollama_data


def print_created_person(index, face_path, deepface_data, ollama_data):
    print(f"  [+] pool/person{index}/ angelegt")
    print(f"      face.jpg: {face_path}")
    print(
        f"      Emotion: {deepface_data['Emotion']}, Alter: {deepface_data['Alter']}, "
        f"Geschlecht: {deepface_data['Geschlecht']}"
    )
    print(f"      Beschreibung: {ollama_data['description'][:80]}...")


def load_original_image_as_face(image_path):
    return Image.open(image_path).convert("RGB")


def create_ai_pool_person(index, image_path, runtime, review_enabled):
    folder = POOL_DIR / f"person{index}"
    if folder.exists():
        print(f"  [!] {folder} existiert bereits, wird uebersprungen.")
        return False

    folder.mkdir(parents=False, exist_ok=False)
    try:
        face_path = folder / "face.jpg"
        try:
            if runtime["face_detector_mode"] == "yolo":
                face_image = create_face_crop(
                    image_path=image_path,
                    yolo_model=runtime["yolo_model"],
                    device=runtime["yolo_device"],
                    confidence=runtime["face_confidence"],
                    padding=DEFAULT_FACE_PADDING,
                    rembg_session=runtime["rembg_session"],
                )
            else:
                face_image = create_face_crop_with_haar(
                    image_path=image_path,
                    detectors=runtime["haar_detectors"],
                    padding=DEFAULT_FACE_PADDING,
                    rembg_session=runtime["rembg_session"],
                )
        except Exception as exc:
            print(f"  [!] Kein stabiler Face-Crop fuer {image_path.name}, nutze Originalbild als Fallback: {exc}")
            face_image = load_original_image_as_face(image_path)
        face_image.save(face_path, format="JPEG", quality=95)

        deepface_settings = runtime["deepface_settings"]
        deepface_data = analyze_with_deepface(
            face_path=face_path,
            use_retinaface=deepface_settings.get("use_retinaface", True),
        )

        source_description = build_source_description(image_path, deepface_data)
        ollama_error = runtime.get("ollama_error")
        if ollama_error:
            print(f"  [!] Ollama-Fallback fuer {image_path.name}: {ollama_error}")
            ollama_data = {
                "prompt": runtime["ollama_settings"]["prompt"],
                "source_prompt": POOL_SOURCE_PROMPT,
                "source_description": source_description,
                "description": source_description,
            }
        else:
            ollama_data = generate_ollama_description(
                source_description=source_description,
                ollama_settings=runtime["ollama_settings"],
            )

        if review_enabled:
            deepface_data, ollama_data = review_generated_data(index, deepface_data, ollama_data)

        write_yaml(folder / "deepface.yaml", deepface_data)
        write_yaml(folder / "ollama.yaml", ollama_data)
        print_created_person(index, face_path, deepface_data, ollama_data)
        return True
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise


def create_random_pool_person(index, description, crime_story, image_path=None, review_enabled=False):
    folder = POOL_DIR / f"person{index}"
    if folder.exists():
        print(f"  [!] {folder} existiert bereits, wird uebersprungen.")
        return False

    folder.mkdir(parents=False, exist_ok=False)
    try:
        deepface_data = build_random_deepface_yaml()
        ollama_data = build_random_ollama_yaml(description, crime_story)

        if image_path:
            image = Image.open(image_path).convert("RGB")
            image.save(folder / "face.jpg", format="JPEG", quality=95)
        else:
            print("  [!] Kein Bild gefunden - face.jpg fehlt. Bitte manuell ersetzen.")

        if review_enabled:
            deepface_data, ollama_data = review_generated_data(index, deepface_data, ollama_data)

        write_yaml(folder / "deepface.yaml", deepface_data)
        write_yaml(folder / "ollama.yaml", ollama_data)
        print_created_person(index, folder / "face.jpg", deepface_data, ollama_data)
        return True
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise


def build_runtime(config, use_images):
    runtime = {
        "config": config,
        "face_confidence": get_face_yolo_confidence(config),
        "deepface_settings": get_deepface_settings(config),
        "ollama_settings": get_ollama_settings(config),
        "yolo_device": None,
        "yolo_model": None,
        "haar_detectors": None,
        "face_detector_mode": None,
        "rembg_session": None,
        "ollama_process": None,
        "ollama_error": None,
    }

    if not use_images:
        return runtime

    runtime["yolo_device"] = get_yolo_device()
    try:
        runtime["yolo_model"] = load_face_yolo_model(runtime["yolo_device"])
        runtime["face_detector_mode"] = "yolo"
    except Exception as exc:
        print(f"  [!] Face-YOLO lokal nicht nutzbar, wechsle auf Haar-Cascade-Fallback: {exc}")
        runtime["haar_detectors"] = load_haar_face_detector()
        runtime["face_detector_mode"] = "haar"
    runtime["rembg_session"] = create_rembg_session()
    if runtime["rembg_session"] is None and new_session is None:
        print("  [!] rembg ist lokal nicht installiert. Gesichtscrops werden ohne Hintergrundentfernung gespeichert.")

    ollama_settings = runtime["ollama_settings"]
    if ollama_settings.get("enabled", True):
        try:
            runtime["ollama_process"] = ensure_ollama_ready(ollama_settings["model"])
        except Exception as exc:
            runtime["ollama_error"] = str(exc)
    return runtime


def close_runtime(runtime):
    stop_process(runtime.get("ollama_process"))


def main():
    parser = argparse.ArgumentParser(description="Pool-Personen Generator fuer Glaeserner-Mensch-SE2")
    parser.add_argument("--count", type=int, default=1, help="Wie viele Pool-Personen anlegen (Standard: 1)")
    parser.add_argument("--image", type=str, default=None, help="Pfad zu einem eigenen Bild (nur bei count=1 sinnvoll)")
    parser.add_argument("--images", nargs="+", help="Mehrere Bildpfade fuer mehrere neue Pool-Personen")
    parser.add_argument("--image-dir", type=str, default=None, help="Ordner mit Bildern fuer mehrere neue Pool-Personen")
    parser.add_argument("--no-review", action="store_true", help="Interaktive Nachbearbeitung nach der Generierung ueberspringen")
    args = parser.parse_args()

    if args.count < 1 or args.count > 50:
        print("[!] --count muss zwischen 1 und 50 liegen.")
        sys.exit(1)

    if args.image and (args.images or args.image_dir):
        print("[!] --image kann nicht mit --images oder --image-dir kombiniert werden.")
        sys.exit(1)

    if args.images and args.image_dir:
        print("[!] --images und --image-dir bitte nicht gleichzeitig verwenden.")
        sys.exit(1)

    if args.count > 1 and args.image:
        print("[!] --image ist nur zusammen mit --count 1 sinnvoll.")
        sys.exit(1)

    config = load_config()
    image_sources = gather_images(image_paths=args.images, image_dir=args.image_dir)

    if args.image:
        single_image = resolve_image_path(args.image)
        if single_image is None:
            print(f"[!] Eigenes Bild nicht gefunden: {args.image}")
            sys.exit(1)
        image_sources = [single_image]

    if args.images and not image_sources:
        print("[!] Keine gueltigen Bilder aus --images gefunden.")
        sys.exit(1)
    if args.image_dir and not image_sources:
        print("[!] Keine gueltigen Bilder im angegebenen Ordner gefunden.")
        sys.exit(1)

    if image_sources:
        args.count = len(image_sources)

    review_enabled = (not args.no_review) and sys.stdin.isatty()
    if not review_enabled and not args.no_review and image_sources:
        print("[i] Keine interaktive Konsole erkannt, Review wird automatisch uebersprungen.")

    print(f"\nPool-Builder: {args.count} Person(en) werden angelegt...\n")

    start_index = get_next_person_index()
    created = 0
    descriptions, crime_stories = build_text_sequences(args.count)
    runtime = build_runtime(config, use_images=bool(image_sources))

    try:
        for offset in range(args.count):
            index = start_index + offset
            try:
                if image_sources:
                    image = image_sources[offset]
                    if create_ai_pool_person(index, image, runtime, review_enabled):
                        created += 1
                else:
                    if create_random_pool_person(
                        index=index,
                        description=descriptions[offset],
                        crime_story=crime_stories[offset],
                        image_path=None,
                        review_enabled=review_enabled,
                    ):
                        created += 1
            except Exception as exc:
                print(f"  [!] person{index} konnte nicht erzeugt werden: {exc}")
    finally:
        close_runtime(runtime)

    print(f"\nFertig: {created} von {args.count} Person(en) angelegt.")
    if review_enabled:
        print("Hinweis: Die erzeugten YAML-Dateien koennen spaeter weiterhin direkt manuell angepasst werden.\n")
    else:
        print("Hinweis: Die erzeugten YAML-Dateien koennen spaeter direkt manuell angepasst werden.\n")


if __name__ == "__main__":
    main()
