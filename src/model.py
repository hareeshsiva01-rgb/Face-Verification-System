import tensorflow as tf
from tensorflow.keras import layers, models


def create_face_model():

    model = models.Sequential([

        # Input: 160 x 160 RGB image
        layers.Input(shape=(160, 160, 3)),

        # First convolution block
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # Second convolution block
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # Third convolution block
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # Convert feature maps into one vector
        layers.Flatten(),

        # Learn a compact representation of the face
        layers.Dense(128, activation="relu"),

        # Final face embedding
        layers.Dense(64)
    ])

    return model


# Create our model
model = create_face_model()

# Display model structure
model.summary()