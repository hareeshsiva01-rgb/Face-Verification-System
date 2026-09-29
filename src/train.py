import tensorflow as tf
import numpy as np
import csv
import os

from siamese_model import create_siamese_model


# --------------------------------------------------
# Settings
# --------------------------------------------------

IMAGE_SIZE = (160, 160)

BATCH_SIZE = 8

EPOCHS = 10


# --------------------------------------------------
# Load image
# --------------------------------------------------

def load_image(image_path):

    image = tf.io.read_file(image_path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    return image


# --------------------------------------------------
# Load CSV
# --------------------------------------------------

def load_pairs(csv_file):

    image1_list = []
    image2_list = []
    labels = []

    with open(csv_file, "r") as file:

        reader = csv.DictReader(file)

        for row in reader:

            image1_list.append(
                row["image1"]
            )

            image2_list.append(
                row["image2"]
            )

            labels.append(
                float(row["label"])
            )

    return (
        image1_list,
        image2_list,
        labels
    )


# --------------------------------------------------
# Create TensorFlow dataset
# --------------------------------------------------

def create_dataset(csv_file):

    image1_paths, image2_paths, labels = load_pairs(
        csv_file
    )

    print()
    print(
        "Loading:",
        csv_file
    )

    print(
        "Pairs:",
        len(labels)
    )

    images1 = []

    images2 = []

    for path in image1_paths:

        images1.append(
            load_image(path)
        )

    for path in image2_paths:

        images2.append(
            load_image(path)
        )

    images1 = np.array(
        images1,
        dtype=np.float32
    )

    images2 = np.array(
        images2,
        dtype=np.float32
    )

    labels = np.array(
        labels,
        dtype=np.float32
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            (
                images1,
                images2
            ),
            labels
        )
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    return dataset


# --------------------------------------------------
# Contrastive Loss
# --------------------------------------------------

def contrastive_loss(
    y_true,
    distance,
    margin=1.0
):

    y_true = tf.cast(
        y_true,
        tf.float32
    )

    same_loss = (
        y_true *
        tf.square(distance)
    )

    different_loss = (
        (1 - y_true)
        *
        tf.square(
            tf.maximum(
                margin - distance,
                0
            )
        )
    )

    loss = (
        same_loss +
        different_loss
    )

    return tf.reduce_mean(loss)


# --------------------------------------------------
# Load training data
# --------------------------------------------------

print()
print("================================")
print("Loading training data")
print("================================")

train_dataset = create_dataset(
    "dataset/train_pairs.csv"
)


# --------------------------------------------------
# Load validation data
# --------------------------------------------------

print()
print("================================")
print("Loading validation data")
print("================================")

validation_dataset = create_dataset(
    "dataset/val_pairs.csv"
)


# --------------------------------------------------
# Create model
# --------------------------------------------------

print()
print("================================")
print("Creating Siamese model")
print("================================")

model = create_siamese_model()


# --------------------------------------------------
# Compile model
# --------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss=contrastive_loss
)


print()
print("Model compiled successfully.")


# --------------------------------------------------
# Train model
# --------------------------------------------------

print()
print("================================")
print("Starting training")
print("================================")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)


# --------------------------------------------------
# Save model
# --------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)

model.save(
    "models/face_verification.keras"
)


print()
print("================================")
print("Training completed")
print("================================")

print(
    "Model saved to:",
    "models/face_verification.keras"
)