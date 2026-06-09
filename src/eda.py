import os
import matplotlib.pyplot as plt

dataset_path = "dataset/train"

classes = [
    d for d in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, d))
]

counts = []

for cls in classes:
    class_path = os.path.join(dataset_path, cls)

    counts.append(
        len([
            f for f in os.listdir(class_path)
            if not f.startswith(".")
        ])
    )

plt.figure(figsize=(12,6))
plt.bar(classes, counts)
plt.xticks(rotation=90)
plt.title("Class Distribution")
plt.tight_layout()

plt.savefig("outputs/class_distribution.png")
plt.show()