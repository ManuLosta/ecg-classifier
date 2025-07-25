import tensorflow as tf
from tensorflow.keras import layers, models


def create_cnn1d_model(input_shape, num_classes):
    """
    Create a CNN1D model with the paper architecture using TensorFlow/Keras.

    Args:
        input_shape: Tuple of (sequence_length, input_channels)
        num_classes: Number of output classes

    Returns:
        A Keras model instance.
    """
    inputs = layers.Input(shape=input_shape)

    # Large block with parallel branches
    branch1 = layers.Conv1D(64, kernel_size=7, padding="same", activation="relu")(
        inputs
    )
    branch2 = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(
        inputs
    )
    branch3 = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(
        inputs
    )
    branch4 = layers.Conv1D(64, kernel_size=1, padding="same", activation="relu")(
        inputs
    )

    # Concatenate all branches
    concatenated = layers.Concatenate()([branch1, branch2, branch3, branch4])

    # Small blocks
    sb1 = layers.Conv1D(64, kernel_size=7, padding="same", activation="relu")(
        concatenated
    )
    sb1 = layers.MaxPooling1D(pool_size=2)(sb1)
    sb1 = layers.Dropout(0.3)(sb1)

    sb2 = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(
        concatenated
    )
    sb2 = layers.MaxPooling1D(pool_size=2)(sb2)
    sb2 = layers.Dropout(0.3)(sb2)

    sb3 = layers.Conv1D(64, kernel_size=1, padding="same", activation="relu")(
        concatenated
    )
    sb3 = layers.MaxPooling1D(pool_size=2)(sb3)
    sb3 = layers.Dropout(0.3)(sb3)

    # Concatenate small block outputs
    merged = layers.Concatenate()([sb1, sb2, sb3])

    # Global average pooling
    pooled = layers.GlobalAveragePooling1D()(merged)

    # Fully connected layers
    dense1 = layers.Dense(64, activation="relu")(pooled)
    dense1 = layers.Dropout(0.3)(dense1)
    outputs = layers.Dense(num_classes, activation="softmax")(dense1)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model
