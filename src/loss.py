import tensorflow as tf


def contrastive_loss(y_true, distance, margin=1.0):

    y_true = tf.cast(y_true, tf.float32)

    same_loss = y_true * tf.square(distance)

    different_loss = (
        (1 - y_true)
        * tf.square(
            tf.maximum(margin - distance, 0)
        )
    )

    loss = same_loss + different_loss

    return tf.reduce_mean(loss)


print("Contrastive loss function created successfully.")