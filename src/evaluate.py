import tensorflow as tf
from preprocess import get_datasets

train_ds, val_ds, test_ds, class_names = get_datasets()

model = tf.keras.models.load_model(
    "models/mobilenet_model.keras"
)

loss, accuracy = model.evaluate(test_ds)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")