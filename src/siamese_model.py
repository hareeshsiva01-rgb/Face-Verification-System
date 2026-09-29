import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, Model

from model import create_face_model


# --------------------------------------------------
# Custom distance layer
# --------------------------------------------------
@keras.saving.register_keras_serializable()
class EuclideanDistance(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):

        embedding1, embedding2 = inputs

        distance = tf.sqrt(
            tf.reduce_sum(
                tf.square(
                    embedding1 - embedding2
                ),
                axis=1,
                keepdims=True
            )
        )

        return distance

    def get_config(self):

        config = super().get_config()

        return config


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
# Create Siamese Model
# --------------------------------------------------

def create_siamese_model():

    face_model = create_face_model()

    face1 = layers.Input(
        shape=(160, 160, 3),
        name="face_1"
    )

    face2 = layers.Input(
        shape=(160, 160, 3),
        name="face_2"
    )

    embedding1 = face_model(face1)

    embedding2 = face_model(face2)

    distance = EuclideanDistance(
        name="euclidean_distance"
    )(
        [embedding1, embedding2]
    )

    siamese = Model(
        inputs=[face1, face2],
        outputs=distance
    )

    return siamese


# --------------------------------------------------
# Create and compile model
# --------------------------------------------------

siamese_model = create_siamese_model()

siamese_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss=contrastive_loss
)


print()
print("Siamese model created successfully.")

print()
print("Siamese model compiled successfully.")

print()

siamese_model.summary()