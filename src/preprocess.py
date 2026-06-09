import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def get_datasets():

    train_ds = tf.keras.utils.image_dataset_from_directory(
        "dataset/train",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        "dataset/validation",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        "dataset/test",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names

    normalization_layer = tf.keras.layers.Rescaling(1./255)

    train_ds = train_ds.map(
        lambda x, y: (normalization_layer(x), y)
    )

    val_ds = val_ds.map(
        lambda x, y: (normalization_layer(x), y)
    )

    test_ds = test_ds.map(
        lambda x, y: (normalization_layer(x), y)
    )

    return train_ds, val_ds, test_ds, class_names