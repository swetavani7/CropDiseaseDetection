import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from preprocess import get_datasets

train_ds, val_ds, test_ds, class_names = get_datasets()

model = tf.keras.models.load_model(
    "models/mobilenet_model.keras"
)

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(images)

    y_true.extend(labels.numpy())

    y_pred.extend(
        np.argmax(predictions, axis=1)
    )

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

cm = confusion_matrix(
    y_true,
    y_pred
)

print(cm)


plt.figure(figsize=(12,10))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix.png")

print("Confusion matrix saved!")

plt.show()