import os
import matplotlib.pyplot as plt
from PIL import Image

dataset_path = "dataset/train"

classes = os.listdir(dataset_path)

plt.figure(figsize=(15,10))

for i, cls in enumerate(classes[:9]):

    img_name = os.listdir(
        os.path.join(dataset_path, cls)
    )[0]

    img_path = os.path.join(
        dataset_path,
        cls,
        img_name
    )

    img = Image.open(img_path)

    plt.subplot(3,3,i+1)
    plt.imshow(img)
    plt.title(cls)
    plt.axis("off")

plt.tight_layout()

plt.savefig("outputs/sample_images.png")
plt.show()