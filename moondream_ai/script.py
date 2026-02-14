import time
import torch
import os
import shutil
from transformers import AutoModelForCausalLM
from PIL import Image

print("--- Loading Model into VRAM... ---")
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda",
)

INPUT_DIR = "/data/input"
PROCESSED_DIR = "/data/processed"

while True:
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not files:
        time.sleep(0.5)
        continue

    for filename in files:
        img_path = os.path.join(INPUT_DIR, filename)
        # We save the text and moved image to the PROCESSED_DIR (which is your 'final' folder)
        dest_path = os.path.join(PROCESSED_DIR, filename)
        txt_path = os.path.join(PROCESSED_DIR, f"{os.path.splitext(filename)[0]}.txt")

        try:
            image = Image.open(img_path).convert("RGB")
            answer = \
            model.query(image, "Describe the person. Gender, Hairstyle, facial features, clothes, activity, objects")[
                "answer"]

            # Write the result to the 'final' folder
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(answer.strip())

            # Move image to 'final' folder
            shutil.move(img_path, dest_path)
            print(f"Processed {filename} successfully.")

        except Exception as e:
            print(f"Error: {e}")