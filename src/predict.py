import tensorflow as tf
import numpy as np

from PIL import Image

IMG_SIZE = (224,224)

class_names = [
    "Apple___Black_rot",
    "Apple___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight"
]

model = tf.keras.models.load_model(
    "models/mobilenet_model.keras"
)

image_path = input(
    "Enter image path: "
)

image = Image.open(image_path)

image = image.resize(
    IMG_SIZE
)

image = np.array(image)/255.0

image = np.expand_dims(
    image,
    axis=0
)

prediction = model.predict(image)

predicted_class = class_names[
    np.argmax(prediction)
]

confidence = np.max(prediction)

print(
    f"Disease: {predicted_class}"
)

print(
    f"Confidence: {confidence*100:.2f}%"
)