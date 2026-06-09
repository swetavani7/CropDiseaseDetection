import tensorflow as tf
from preprocess import get_datasets

# Load datasets
train_ds, val_ds, test_ds, class_names = get_datasets()

num_classes = len(class_names)

print("Classes:", class_names)
print("Number of Classes:", num_classes)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = tf.keras.Sequential([

    base_model,

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        128,
        activation='relu'
    ),

    tf.keras.layers.Dense(
        num_classes,
        activation='softmax'
    )
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop]
)

import matplotlib.pyplot as plt

# Accuracy Graph
plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Transfer Learning Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend([
    'Train',
    'Validation'
])

plt.savefig(
    'outputs/transfer_accuracy.png'
)

plt.show()

# Loss Graph
plt.figure(figsize=(8,5))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Transfer Learning Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend([
    'Train',
    'Validation'
])

plt.savefig(
    'outputs/transfer_loss.png'
)

plt.show()

model.save("models/mobilenet_model.keras")

print("Transfer model saved!")


    