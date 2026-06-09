import os
import random
import shutil

SOURCE_DIR = "PlantVillage"
TARGET_DIR = "dataset"

train_ratio = 0.7
val_ratio = 0.15

for class_name in os.listdir(SOURCE_DIR):

    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    train_end = int(len(images) * train_ratio)
    val_end = int(len(images) * (train_ratio + val_ratio))

    splits = {
        "train": images[:train_end],
        "validation": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split_name, split_images in splits.items():

        target_class_dir = os.path.join(
            TARGET_DIR,
            split_name,
            class_name
        )

        os.makedirs(target_class_dir, exist_ok=True)

        for img in split_images:
            shutil.copy(
                os.path.join(class_path, img),
                os.path.join(target_class_dir, img)
            )

print("Dataset split completed!")