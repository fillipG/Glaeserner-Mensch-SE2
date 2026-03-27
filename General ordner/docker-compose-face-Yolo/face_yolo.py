"""
Face Detection und Hintergrund-Entfernung mit YOLO

Erkennt Gesichter in Eingabebildern mittels YOLOv8n-face, entfernt den Hintergrund
und verteilt die extrahierten Gesichter an spezialisierte Inbox-Ordner
fuer Deepface- und Moondream-Analyse.

Zustaendigkeiten:
- Ueberwacht Input-Verzeichnis dauerhaft auf neue Bilder
- Fuehrt Bildvorverarbeitung durch (Aufhellung, CLAHE-Kontrast, Schaerfung)
- Fuehrt YOLO-Gesichtserkennung mit konfigurierbarer Konfidenz durch
- Entfernt Hintergruende mit rembg
- Verteilt Gesichter an sketch/, deepface_inbox/, moondream_inbox/
- Speichert Debug-Informationen bei 0-Gesichter-Ergebnissen
"""

import os
import time
from fnmatch import fnmatch

import cv2
import numpy as np
import yaml
from PIL import Image
from rembg import new_session, remove
from ultralytics import YOLO


INPUT_DIR = "main_image"
YAML_PATH = "final/faces_log.yaml"
SKETCH_DIR = "sketch"
DEEPFACE_INBOX = "deepface_inbox"
MOONDREAM_INBOX = "moondream_inbox"
CONFIG_PATH = "config.yaml"
DEBUG_DIR = "debug_face_yolo"
DEBUG_BODY_DIR = "debug_body"
FACE_YOLO_WEIGHTS = "yolov8n-face.pt"
BODY_YOLO_WEIGHTS = "/app/yolov8n.pt"
SEG_YOLO_WEIGHTS = "/app/yolov8n-seg.pt"

DEFAULT_CONFIDENCE = 0.5
DEFAULT_PADDING = 40
DEFAULT_MAX_FACES = 4
DEFAULT_MOONDREAM_CROP_MODE = "face"
DEFAULT_BODY_CONFIDENCE = 0.35
DEFAULT_BODY_PADDING_RATIO = 0.12
DEFAULT_BODY_FALLBACK_TO_FACE = True
DEFAULT_BODY_MATCHING_REQUIRED = False
DEFAULT_DEBUG_MATCHING = True


def load_runtime_config():
    """
    Laedt aktuelle Laufzeitwerte aus config.yaml.

    Ermoeglicht Live-Anpassungen ohne Neustart. Liest Face-YOLO-, Body-YOLO-
    und Moondream-Crop-Einstellungen ein. Bei Lesefehler oder fehlender
    Konfiguration: Rueckfall auf globale Standardwerte.
    """
    runtime = {
        "confidence": DEFAULT_CONFIDENCE,
        "max_faces": DEFAULT_MAX_FACES,
        "moondream_crop_mode": DEFAULT_MOONDREAM_CROP_MODE,
        "body_confidence": DEFAULT_BODY_CONFIDENCE,
        "body_padding_ratio": DEFAULT_BODY_PADDING_RATIO,
        "body_fallback_to_face": DEFAULT_BODY_FALLBACK_TO_FACE,
        "body_matching_required": DEFAULT_BODY_MATCHING_REQUIRED,
        "debug_matching": DEFAULT_DEBUG_MATCHING,
    }

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
    except Exception as exc:
        print(f"  Konnte Face-YOLO Config nicht lesen, nutze Standardwerte: {exc}")
        return runtime

    face_yolo_cfg = config.get("face_yolo", {})
    if not isinstance(face_yolo_cfg, dict):
        face_yolo_cfg = {}

    try:
        runtime["confidence"] = max(0.10, min(0.90, float(face_yolo_cfg.get("confidence", DEFAULT_CONFIDENCE))))
    except (TypeError, ValueError):
        runtime["confidence"] = DEFAULT_CONFIDENCE

    try:
        runtime["max_faces"] = max(1, min(4, int(face_yolo_cfg.get("max_faces", DEFAULT_MAX_FACES))))
    except (TypeError, ValueError):
        runtime["max_faces"] = DEFAULT_MAX_FACES

    mode = str(face_yolo_cfg.get("moondream_crop_mode", DEFAULT_MOONDREAM_CROP_MODE)).strip().lower()
    if mode not in {"face", "body", "body_seg", "shadow"}:
        mode = DEFAULT_MOONDREAM_CROP_MODE
    runtime["moondream_crop_mode"] = mode

    try:
        runtime["body_confidence"] = max(
            0.10,
            min(0.90, float(face_yolo_cfg.get("body_confidence", DEFAULT_BODY_CONFIDENCE))),
        )
    except (TypeError, ValueError):
        runtime["body_confidence"] = DEFAULT_BODY_CONFIDENCE

    try:
        runtime["body_padding_ratio"] = max(
            0.0,
            min(0.5, float(face_yolo_cfg.get("body_padding_ratio", DEFAULT_BODY_PADDING_RATIO))),
        )
    except (TypeError, ValueError):
        runtime["body_padding_ratio"] = DEFAULT_BODY_PADDING_RATIO

    runtime["body_fallback_to_face"] = bool(
        face_yolo_cfg.get("body_fallback_to_face", DEFAULT_BODY_FALLBACK_TO_FACE)
    )
    runtime["body_matching_required"] = bool(
        face_yolo_cfg.get("body_matching_required", DEFAULT_BODY_MATCHING_REQUIRED)
    )
    runtime["debug_matching"] = bool(face_yolo_cfg.get("debug_matching", DEFAULT_DEBUG_MATCHING))
    return runtime


def cleanup_previous_batch():
    """
    Loescht veraltete Gesichts-Dateien aus vorherigen Laeufen.

    Entfernt alte face*- und batch*_face*-Artefakte aus sketch/, deepface_inbox/,
    moondream_inbox/ und final/, damit kein Lauf mit alten Dateien kollidiert.
    """
    targets = {
        SKETCH_DIR: ["face*.png", "face*.jpg", "face*.jpeg", "batch*_face*.png", "batch*_face*.jpg", "batch*_face*.jpeg"],
        DEEPFACE_INBOX: ["face*.png", "face*.jpg", "face*.jpeg", "batch*_face*.png", "batch*_face*.jpg", "batch*_face*.jpeg"],
        MOONDREAM_INBOX: ["face*.png", "face*.jpg", "face*.jpeg", "batch*_face*.png", "batch*_face*.jpg", "batch*_face*.jpeg"],
        "final": ["face*_*.yaml", "batch*_face*_*.yaml", "faces_log.yaml"],
        DEBUG_BODY_DIR: [
            "face*_body.png",
            "face*_body_seg.png",
            "face*_moondream.png",
            "batch*_face*_body.png",
            "batch*_face*_body_seg.png",
            "batch*_face*_moondream.png",
            "matching_debug.yaml",
        ],
    }

    for folder, patterns in targets.items():
        os.makedirs(folder, exist_ok=True)
        for name in os.listdir(folder):
            if any(fnmatch(name.lower(), pattern.lower()) for pattern in patterns):
                file_path = os.path.join(folder, name)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as exc:
                    print(f"  Konnte altes Artefakt nicht loeschen: {file_path} ({exc})")


def preprocess_for_detection(image):
    """
    Verbessert Eingabebild fuer robustere Gesichtserkennung.

    Wendet Aufhellung, CLAHE-Kontrastverstaerkung und Schaerfungsfilter an.
    Gibt ein verbessertes BGR-Bild (numpy.ndarray) zurueck.
    """
    brightened = cv2.convertScaleAbs(image, alpha=1.08, beta=8)
    lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(enhanced, -1, sharpen_kernel)


def build_batch_id():
    """
    Erzeugt eine eindeutige Batch-ID fuer einen neuen Foto-Durchlauf.

    Die ID wird in Datei- und Personen-IDs eingebaut, damit spaete Ergebnisse
    alter Durchlaeufe nicht mehr mit dem aktuellen Batch verwechselt werden.
    """
    return f"batch{int(time.time() * 1000)}"


def write_debug_output(image_name, original_image, enhanced_image, current_confidence, results):
    """
    Speichert Debug-Informationen bei Erkennungsfehlern (0 Gesichter).

    Schreibt Original- und verbessertes Bild sowie Konfidenz- und
    Helligkeitsmetriken in debug_face_yolo/ zur spaeteren Analyse.
    """
    os.makedirs(DEBUG_DIR, exist_ok=True)

    original_debug_path = os.path.join(DEBUG_DIR, "last_input.jpg")
    enhanced_debug_path = os.path.join(DEBUG_DIR, "last_enhanced.jpg")
    cv2.imwrite(original_debug_path, original_image)
    cv2.imwrite(enhanced_debug_path, enhanced_image)

    boxes = results[0].boxes if results and len(results) > 0 else None
    confidences = []
    if boxes is not None and boxes.conf is not None:
        confidences = [float(value) for value in boxes.conf.cpu().tolist()]

    mean_brightness = float(cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY).mean())
    max_conf = max(confidences) if confidences else 0.0
    print(f"  [DEBUG] Kein Gesicht erkannt fuer {image_name}")
    print(
        f"  [DEBUG] confidence={current_confidence:.2f}, "
        f"max_box_conf={max_conf:.3f}, brightness={mean_brightness:.1f}"
    )
    print(f"  [DEBUG] Debug-Bilder gespeichert: {original_debug_path}, {enhanced_debug_path}")


def _clamp_box(x1, y1, x2, y2, width, height):
    return (
        max(0, int(x1)),
        max(0, int(y1)),
        min(width, int(x2)),
        min(height, int(y2)),
    )


def _measure_sharpness(image, x1, y1, x2, y2):
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_best_face_candidates(results, image, face_limit):
    height, width = image.shape[:2]
    diagonal = max(1.0, float(np.hypot(width, height)))
    frame_area = max(1.0, float(width * height))
    center_x = width / 2.0
    center_y = height / 2.0
    candidates = []

    # Bewertet jede Face-Box fuer die spaetere Auswahl.
    # Kombiniert dafuer confidence, Flaeche, Bildzentrum und Schaerfe.
    # Bevorzugt damit grosse, scharfe und zentral platzierte Gesichter.
    # Drosselt so unbrauchbare Neben-Treffer trotz hoher YOLO-Confidence.
    for result in results:
        for box in result.boxes:
            # Extrahiert Bounding-Box-Koordinaten.
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, width, height)
            if x2 <= x1 or y2 <= y1:
                continue

            area_score = ((x2 - x1) * (y2 - y1)) / frame_area
            confidence_score = float(box.conf[0].cpu().item())
            candidate_center_x = (x1 + x2) / 2.0
            candidate_center_y = (y1 + y2) / 2.0
            center_distance = float(np.hypot(candidate_center_x - center_x, candidate_center_y - center_y))
            center_score = max(0.0, 1.0 - (center_distance / diagonal))
            sharpness_score = min(_measure_sharpness(image, x1, y1, x2, y2) / 800.0, 1.0)
            overall_score = (
                0.45 * confidence_score
                + 0.30 * area_score
                + 0.15 * center_score
                + 0.10 * sharpness_score
            )

            candidates.append(
                {
                    "box": (x1, y1, x2, y2),
                    "confidence": confidence_score,
                    "area_score": area_score,
                    "center_score": center_score,
                    "sharpness_score": sharpness_score,
                    "overall_score": overall_score,
                }
            )

    candidates.sort(
        key=lambda candidate: (
            candidate["overall_score"],
            candidate["confidence"],
            candidate["area_score"],
        ),
        reverse=True,
    )
    selected = candidates[:face_limit]
    # Sortiert Gewinner nach der Qualitaetsauswahl stabil von links nach rechts.
    # Haelt damit face1/face2/... zwischen den nachgelagerten Modellen konsistent.
    selected.sort(key=lambda candidate: (candidate["box"][0], candidate["box"][1]))
    return selected, candidates


def detect_persons(model, image, confidence, device):
    """
    Erkennt Personen mit dem Standard-YOLO-Modell.
    Es wird nur die Klasse "person" ausgewertet.
    """
    if model is None:
        return []

    results = model(image, conf=confidence, device=device, classes=[0], verbose=False)
    candidates = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            candidates.append(
                {
                    "box": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": float(box.conf[0].cpu().item()),
                }
            )
    return candidates


def match_faces_to_persons(selected_faces, persons, image_shape):
    """
    Ordnet ausgewaehlte Gesichter passenden Personenboxen zu.
    Das Gesichtszentrum muss innerhalb der Personenbox liegen.
    """
    matches = []
    used_persons = set()
    image_area = max(1.0, float(image_shape[0] * image_shape[1]))

    for face in selected_faces:
        fx1, fy1, fx2, fy2 = face["box"]
        face_center_x = (fx1 + fx2) / 2.0
        face_center_y = (fy1 + fy2) / 2.0

        best_idx = None
        best_score = -1.0

        for idx, person in enumerate(persons):
            if idx in used_persons:
                continue

            px1, py1, px2, py2 = person["box"]
            # Prueft, ob das Gesichtszentrum innerhalb der Personenbox liegt.
            # Verhindert so offensichtliche Fehlzuordnungen bei mehreren Personen.
            if not (px1 <= face_center_x <= px2 and py1 <= face_center_y <= py2):
                continue

            person_area = max(1, (px2 - px1) * (py2 - py1))
            # Bevorzugt Personenboxen, deren Kopfzone vertikal zum Gesicht passt.
            # Gewichtet groessere Boxen nur leicht als Zusatzsignal.
            dist_y = abs(face_center_y - (py1 + (py2 - py1) * 0.28))
            center_bonus = 1.0 / (1.0 + dist_y)
            size_bonus = min(person_area / image_area, 1.0)
            score = 0.7 * center_bonus + 0.3 * size_bonus
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            used_persons.add(best_idx)
            matches.append(
                {
                    "face_id": face["face_id"],
                    "face_box": face["box"],
                    "person_box": persons[best_idx]["box"],
                    "matched": True,
                    "match_score": round(best_score, 3),
                    "fallback_reason": None,
                }
            )
        else:
            matches.append(
                {
                    "face_id": face["face_id"],
                    "face_box": face["box"],
                    "person_box": None,
                    "matched": False,
                    "match_score": None,
                    "fallback_reason": "no_person_match",
                }
            )

    return matches


def crop_with_padding(image, box, padding_px=None, padding_ratio=None):
    """
    Schneidet einen Bildausschnitt mit festem oder relativem Rand aus.
    Der Crop wird immer innerhalb der Bildgrenzen begrenzt.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box

    if padding_ratio is not None:
        pad_x = int((x2 - x1) * padding_ratio)
        pad_y = int((y2 - y1) * padding_ratio)
    else:
        pad_x = DEFAULT_PADDING if padding_px is None else int(padding_px)
        pad_y = DEFAULT_PADDING if padding_px is None else int(padding_px)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()


def remove_background_from_crop(crop_array, session):
    """
    Entfernt den Hintergrund eines Crops ueber die bestehende rembg-Session.
    """
    # Konvertiert BGR zu RGB fuer PIL-Verarbeitung.
    crop_rgb = cv2.cvtColor(crop_array, cv2.COLOR_BGR2RGB)
    crop_pil = Image.fromarray(crop_rgb)
    return remove(crop_pil, session=session)


def save_image(image_obj, output_path):
    """
    Speichert ein PIL- oder OpenCV-Bild im PNG-Zielformat.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if isinstance(image_obj, Image.Image):
        image_obj.save(output_path)
        return
    image_rgb = cv2.cvtColor(image_obj, cv2.COLOR_BGR2RGB)
    Image.fromarray(image_rgb).save(output_path)


def is_valid_body_crop(crop_array):
    """
    Prueft, ob ein Body-Crop sinnvolle Groesse und Proportionen hat.
    """
    h, w = crop_array.shape[:2]
    if h < 120 or w < 60:
        return False
    aspect = h / max(w, 1)
    # Akzeptiert bewusst auch breitere Koerperboxen, weil Moondream mit etwas
    # mehr Hintergrund meist besser umgehen kann als mit zu haeufigem Face-Fallback.
    return 0.8 <= aspect <= 5.0


def build_body_crop(image, person_box, face_box, padding_ratio):
    """
    Schneidet einen robusteren Body-Crop aus der Personenbox.

    Nutzt die volle vertikale Personenhoehe, zentriert den Zuschnitt aber
    horizontal um das Gesicht. So bleiben Koerper und Kleidung erhalten,
    waehrend seitlicher Leerraum bei breiten YOLO-Boxen reduziert wird.
    """
    height, width = image.shape[:2]
    px1, py1, px2, py2 = person_box
    fx1, fy1, fx2, fy2 = face_box

    person_h = py2 - py1
    person_w = px2 - px1
    if person_h <= 0 or person_w <= 0:
        return None

    pad_y = int(person_h * padding_ratio)
    y1 = max(0, py1 - pad_y)
    y2 = min(height, py2 + pad_y)

    target_h = max(1, y2 - y1)
    face_center_x = (fx1 + fx2) / 2.0

    # Bevorzugt einen hochkantigen Ausschnitt fuer Moondream, ohne den Crop
    # kuenstlich enger als das Gesicht plus Oberkoerper werden zu lassen.
    preferred_width = int(target_h / 1.15)
    min_width = int((fx2 - fx1) * 3.2)
    target_w = min(person_w + int(person_w * padding_ratio * 0.5), max(preferred_width, min_width))
    target_w = max(min(target_w, width), fx2 - fx1)

    x1 = int(round(face_center_x - target_w / 2.0))
    x2 = x1 + target_w

    if x1 < 0:
        x2 = min(width, x2 - x1)
        x1 = 0
    if x2 > width:
        shift = x2 - width
        x1 = max(0, x1 - shift)
        x2 = width

    # Schneidet nie schmaler als die eigentliche Personenbox zu, wenn diese bereits
    # schmal genug ist. So bleiben auch Schultern und Arme erhalten.
    x1 = max(0, min(x1, px1))
    x2 = min(width, max(x2, px2 if person_w <= target_w else x2))

    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()


def build_seg_crop(
    image,
    face_box,
    seg_model,
    device,
    fallback_to_face,
    body_confidence=0.35,
    body_padding_ratio=0.12,
):
    """
    Schneidet die Person pixelgenau mit yolov8n-seg.pt aus.
    Hintergrund wird weiss gesetzt, damit Moondream moeglichst wenig Stoerpixel sieht.
    """
    if seg_model is None:
        if fallback_to_face:
            return crop_with_padding(image, face_box), "face_fallback_no_seg_model"
        return None, "none"

    try:
        results = seg_model(image, conf=body_confidence, device=device, classes=[0], verbose=False)

        fx1, fy1, fx2, fy2 = face_box
        face_center_x = (fx1 + fx2) / 2.0
        face_center_y = (fy1 + fy2) / 2.0

        best_mask = None
        best_box = None

        # Waehlt gezielt die Segmentmaske der Person mit dem erkannten Gesicht.
        # Verhindert so, dass versehentlich eine andere Person ausgeschnitten wird.
        for result in results:
            if result.masks is None or result.boxes is None:
                continue
            for index, box in enumerate(result.boxes):
                bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                if not (bx1 <= face_center_x <= bx2 and by1 <= face_center_y <= by2):
                    continue
                best_mask = result.masks.data[index].cpu().numpy()
                best_box = (int(bx1), int(by1), int(bx2), int(by2))
                break
            if best_mask is not None:
                break

        if best_mask is None or best_box is None:
            if fallback_to_face:
                return crop_with_padding(image, face_box), "face_fallback_no_seg_match"
            return None, "none"

        height, width = image.shape[:2]
        mask_resized = cv2.resize(best_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask_resized > 0.5

        result_image = image.copy()
        # Setzt Hintergrund bewusst weiss statt transparent oder schwarz.
        # Lenkt Moondream damit auf Kleidung, Haltung und Koerperform der Zielperson.
        result_image[~mask_bool] = 255

        bx1, by1, bx2, by2 = best_box
        pad_x = int((bx2 - bx1) * body_padding_ratio)
        pad_y = int((by2 - by1) * body_padding_ratio)
        bx1 = max(0, bx1 - pad_x)
        by1 = max(0, by1 - pad_y)
        bx2 = min(width, bx2 + pad_x)
        by2 = min(height, by2 + pad_y)

        seg_crop = result_image[by1:by2, bx1:bx2]
        if seg_crop.size == 0:
            if fallback_to_face:
                return crop_with_padding(image, face_box), "face_fallback_empty_crop"
            return None, "none"

        return seg_crop, "body_seg"
    except Exception as exc:
        print(f"  [SEG-YOLO] Fehler bei Segmentation: {exc}")
        if fallback_to_face:
            return crop_with_padding(image, face_box), "face_fallback_seg_error"
        return None, "none"


def build_moondream_crop(
    image,
    match,
    mode,
    fallback_to_face,
    body_confidence,
    body_padding_ratio,
    body_matching_required,
    seg_model=None,
    device="cuda",
):
    """
    Erzeugt das Bild, das an Moondream weitergegeben wird.
    Je nach Modus wird Face, Body-Box oder Segmentation verwendet.
    """
    # Buendelt die Crop-Auswahl fuer alle Moondream-Modi an einer Stelle.
    # Waehlt je nach Modus Face-, Body- oder Segmentation-Crop und Fallback.
    if mode in ("face", "shadow"):
        match["moondream_source"] = "face"
        if match["fallback_reason"] is None:
            match["fallback_reason"] = "face_mode"
        return crop_with_padding(image, match["face_box"]), "face"

    if mode == "body_seg":
        seg_crop, seg_source = build_seg_crop(
            image,
            match["face_box"],
            seg_model,
            device,
            fallback_to_face,
            body_confidence=body_confidence,
            body_padding_ratio=body_padding_ratio,
        )
        match["moondream_source"] = seg_source
        if seg_source == "body_seg":
            match["fallback_reason"] = None
        elif seg_source.startswith("face_fallback"):
            match["fallback_reason"] = seg_source
        if seg_crop is not None:
            return seg_crop, seg_source
        match["fallback_reason"] = "seg_failed"
        if fallback_to_face:
            return crop_with_padding(image, match["face_box"]), "face"
        return None, "none"

    if mode == "body" and match["matched"] and match["person_box"] is not None:
        body_crop = build_body_crop(
            image,
            match["person_box"],
            match["face_box"],
            body_padding_ratio,
        )
        if body_crop is not None and is_valid_body_crop(body_crop):
            match["moondream_source"] = "body"
            return body_crop, "body"
        match["fallback_reason"] = "invalid_body_crop"

    # Erzwingt bei aktivem Matching-Schalter einen echten Personen-Treffer.
    # Unterbindet damit bewusst den sonst moeglichen Face-Fallback.
    if body_matching_required and not match["matched"]:
        match["fallback_reason"] = "body_required_no_match"
        match["moondream_source"] = "none"
        return None, "none"

    if fallback_to_face:
        if match["fallback_reason"] is None:
            match["fallback_reason"] = "face_fallback"
        match["moondream_source"] = "face"
        return crop_with_padding(image, match["face_box"]), "face"

    match["moondream_source"] = "none"
    return None, "none"

# ========== INITIALISIERUNG ==========
# Laedt YOLO-Modelle einmalig vor Hauptschleife
device = "cuda"
face_model = YOLO(FACE_YOLO_WEIGHTS)
face_model.to(device)

# Laedt Body-YOLO nur fuer Body-Crops und Matching.
person_model = None
if os.path.exists(BODY_YOLO_WEIGHTS):
    person_model = YOLO(BODY_YOLO_WEIGHTS)
    person_model.to(device)
    print(f"[BODY-YOLO] Person-Modell geladen: {BODY_YOLO_WEIGHTS}")
else:
    print("[BODY-YOLO] Gewicht nicht gefunden, Body-Matching deaktiviert.")

# Laedt Segmentation-Modell nur fuer body_seg-Modus.
seg_model = None
if os.path.exists(SEG_YOLO_WEIGHTS):
    seg_model = YOLO(SEG_YOLO_WEIGHTS)
    seg_model.to(device)
    print(f"[SEG-YOLO] Segmentation-Modell geladen: {SEG_YOLO_WEIGHTS}")
else:
    print("[SEG-YOLO] Gewicht nicht gefunden, body_seg nicht verfuegbar.")

# Erstellt rembg-Session fuer Hintergrundentfernung (GPU, Fallback auf CPU)
rembg_session = new_session("u2net")

last_runtime = {}

# ========== HAUPTSCHLEIFE ==========
# Ueberwacht INPUT_DIR dauerhaft auf neue Bilder
print("Warte auf Bilder im Ordner 'main_image'...")

while True:
    image_files = sorted(
        [name for name in os.listdir(INPUT_DIR) if name.endswith((".png", ".jpg", ".jpeg"))]
    )

    if image_files:
        image_name = image_files[0]
        image_path = os.path.join(INPUT_DIR, image_name)
        batch_id = build_batch_id()
        print(f"Bild gefunden: {image_path} - wird verarbeitet...")

        cleanup_previous_batch()

        image = cv2.imread(image_path)
        if image is None:
            print(f"Bild konnte nicht geladen werden: {image_path}")
            time.sleep(0.5)
            continue

        runtime = load_runtime_config()
        if last_runtime.get("confidence") != runtime["confidence"]:
            print(f"  Face-YOLO confidence aktiv: {runtime['confidence']:.2f}")
        if last_runtime.get("max_faces") != runtime["max_faces"]:
            print(f"  Face-YOLO max_faces aktiv: {runtime['max_faces']}")
        if last_runtime.get("moondream_crop_mode") != runtime["moondream_crop_mode"]:
            print(f"  Moondream crop mode aktiv: {runtime['moondream_crop_mode']}")
        last_runtime = runtime.copy()

        detection_image = preprocess_for_detection(image)
        results = face_model(detection_image, conf=runtime["confidence"], device=device)

        selected_candidates, all_candidates = select_best_face_candidates(
            results,
            image,
            runtime["max_faces"],
        )
        raw_face_count = len(all_candidates)
        selected_faces = []
        for face_nr, candidate in enumerate(selected_candidates, start=1):
            selected_face = dict(candidate)
            selected_face["face_id"] = f"{batch_id}_face{face_nr}"
            selected_faces.append(selected_face)

        face_count = len(selected_faces)
        print(f"  Erkannte Gesichter: {raw_face_count}")
        if raw_face_count == 0:
            write_debug_output(image_name, image, detection_image, runtime["confidence"], results)
        elif raw_face_count > face_count:
            print(f"  Begrenze Verarbeitung auf die besten {face_count} Gesichter.")

        if runtime["moondream_crop_mode"] == "face":
            matches = [
                {
                    "face_id": face["face_id"],
                    "face_box": face["box"],
                    "person_box": None,
                    "matched": False,
                    "match_score": None,
                    "fallback_reason": "face_mode",
                    "moondream_source": "face",
                }
                for face in selected_faces
            ]
        else:
            persons = detect_persons(person_model, image, runtime["body_confidence"], device)
            matches = match_faces_to_persons(selected_faces, persons, image.shape[:2])
            print(f"  Erkannte Personen: {len(persons)}")

        os.makedirs(os.path.dirname(YAML_PATH), exist_ok=True)
        with open(YAML_PATH, "w", encoding="utf-8") as yaml_file:
            yaml.safe_dump(
                {
                    "batch_id": batch_id,
                    "face_count": face_count,
                },
                yaml_file,
                sort_keys=False,
                allow_unicode=True,
            )

        for folder in [SKETCH_DIR, DEEPFACE_INBOX, MOONDREAM_INBOX]:
            os.makedirs(folder, exist_ok=True)

        # Verarbeitet jedes erkannte Gesicht einzeln.
        for match, candidate in zip(matches, selected_faces):
            face_id = match["face_id"]
            # Erweitert Bounding-Box um Padding mit Bildgrenzen-Clipping.
            face_crop = crop_with_padding(image, match["face_box"])
            if face_crop is None:
                print(f"  Ueberspringe {face_id}: ungueltiger Face-Crop.")
                continue

            # --- GESICHT AN INBOXEN VERTEILEN ---
            face_no_bg = remove_background_from_crop(face_crop, rembg_session)
            # Speichert fuer GUI-Skizze.
            save_image(face_no_bg, os.path.join(SKETCH_DIR, f"{face_id}.png"))
            # Speichert fuer Deepface-Analyse (wird nach Verarbeitung geloescht).
            save_image(face_no_bg, os.path.join(DEEPFACE_INBOX, f"{face_id}.png"))

            moondream_crop, moondream_source = build_moondream_crop(
                image,
                match,
                runtime["moondream_crop_mode"],
                runtime["body_fallback_to_face"],
                runtime["body_confidence"],
                runtime["body_padding_ratio"],
                runtime["body_matching_required"],
                seg_model=seg_model,
                device=device,
            )
            moondream_output = None
            if moondream_crop is None:
                print(f"  Kein Moondream-Crop fuer {face_id}, Fallback deaktiviert.")
            elif moondream_source in {"body", "body_seg"}:
                moondream_output = moondream_crop
                # Speichert fuer Moondream-Analyse (wird nach Verarbeitung geloescht).
                save_image(moondream_output, os.path.join(MOONDREAM_INBOX, f"{face_id}.png"))
            else:
                moondream_output = face_no_bg
                # Speichert fuer Moondream-Analyse (wird nach Verarbeitung geloescht).
                save_image(moondream_output, os.path.join(MOONDREAM_INBOX, f"{face_id}.png"))

            if runtime["moondream_crop_mode"] != "face" and moondream_output is not None:
                os.makedirs(DEBUG_BODY_DIR, exist_ok=True)
                save_image(moondream_output, os.path.join(DEBUG_BODY_DIR, f"{face_id}_moondream.png"))
                if moondream_source == "body_seg":
                    save_image(moondream_output, os.path.join(DEBUG_BODY_DIR, f"{face_id}_body_seg.png"))

            if runtime["moondream_crop_mode"] == "shadow" and match["matched"] and match["person_box"] is not None:
                body_crop = build_body_crop(
                    image,
                    match["person_box"],
                    match["face_box"],
                    runtime["body_padding_ratio"],
                )
                if body_crop is not None:
                    save_image(body_crop, os.path.join(DEBUG_BODY_DIR, f"{face_id}_body.png"))
                seg_debug_crop, seg_debug_source = build_seg_crop(
                    image,
                    match["face_box"],
                    seg_model,
                    device,
                    fallback_to_face=False,
                    body_confidence=runtime["body_confidence"],
                    body_padding_ratio=runtime["body_padding_ratio"],
                )
                if seg_debug_crop is not None and seg_debug_source == "body_seg":
                    save_image(seg_debug_crop, os.path.join(DEBUG_BODY_DIR, f"{face_id}_body_seg.png"))

            print(
                f"  {face_id} verteilt "
                f"(score={candidate['overall_score']:.3f}, conf={candidate['confidence']:.3f}, "
                f"moondream={match.get('moondream_source', moondream_source)})."
            )

        if runtime["debug_matching"] and runtime["moondream_crop_mode"] != "face":
            os.makedirs(DEBUG_BODY_DIR, exist_ok=True)
            debug_data = {
                "image": image_name,
                "mode": runtime["moondream_crop_mode"],
                "face_count": len(matches),
                "body_matching_required": runtime["body_matching_required"],
                "matches": matches,
            }
            with open(os.path.join(DEBUG_BODY_DIR, "matching_debug.yaml"), "w", encoding="utf-8") as debug_file:
                yaml.safe_dump(debug_data, debug_file, sort_keys=False, allow_unicode=True)

        # Loescht Originalbild nach erfolgreicher Verarbeitung.
        try:
            os.remove(image_path)
            print(f"Fertig! Originalbild {image_name} geloescht. Warte auf Analyse...")
        except Exception as exc:
            print(f"Fehler beim Loeschen des Originalbilds: {exc}")

    # Wartet kurz vor naechstem Scan-Durchlauf.
    time.sleep(0.5)
