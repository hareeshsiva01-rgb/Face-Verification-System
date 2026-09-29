import tensorflow as tf
import numpy as np
import csv


IMAGE_SIZE = (160, 160)


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
# Load validation pairs
# --------------------------------------------------

def load_pairs():

    image1_list = []
    image2_list = []
    labels = []

    with open(
        "dataset/val_pairs.csv",
        "r"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            image1_list.append(
                row["image1"]
            )

            image2_list.append(
                row["image2"]
            )

            labels.append(
                int(row["label"])
            )

    return (
        image1_list,
        image2_list,
        labels
    )


# --------------------------------------------------
# Load model
# --------------------------------------------------

print()
print("Loading trained model...")

model = tf.keras.models.load_model(
    "models/face_verification.keras",
    compile=False
)

print("Model loaded successfully.")


# --------------------------------------------------
# Load validation data
# --------------------------------------------------

image1_paths, image2_paths, labels = load_pairs()

print()
print(
    "Validation pairs:",
    len(labels)
)


# --------------------------------------------------
# Prepare images
# --------------------------------------------------

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
    labels
)


# --------------------------------------------------
# Predict distances
# --------------------------------------------------

print()
print("Calculating face distances...")

distances = model.predict(
    [images1, images2],
    verbose=0
)


distances = distances.flatten()


# --------------------------------------------------
# Display sample distances
# --------------------------------------------------

print()
print("Sample results:")

for i in range(
    min(10, len(labels))
):

    print(
        "Label:",
        labels[i],
        "| Distance:",
        round(
            float(distances[i]),
            4
        )
    )


# --------------------------------------------------
# Separate same and different pairs
# --------------------------------------------------

same_distances = distances[
    labels == 1
]

different_distances = distances[
    labels == 0
]


print()
print("Same-person pairs:", len(same_distances))

print(
    "Different-person pairs:",
    len(different_distances)
)


print()
print(
    "Average same-person distance:",
    round(
        float(np.mean(same_distances)),
        4
    )
)

print(
    "Average different-person distance:",
    round(
        float(np.mean(different_distances)),
        4
    )
)