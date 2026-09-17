"""
Model Architecture Builder
==========================
Constructs EfficientNet transfer learning architecture with a custom classification head.
Binary output:
0 -> Real
1 -> AI-Generated
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple

def build_efficientnet_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    variant: str = "B0",
    trainable_base: bool = False,
    learning_rate: float = 1e-4
) -> models.Model:
    """
    Builds and compiles the EfficientNet binary classification model.

    Args:
        input_shape: Image dimensions (height, width, channels), default (224, 224, 3)
        variant: EfficientNet variant ('B0', 'B1', 'B2', etc.)
        trainable_base: Whether to freeze backbone weights for feature extraction
        learning_rate: Initial Adam learning rate
    """
    # Select backbone
    backbone_cls = getattr(tf.keras.applications, f"EfficientNet{variant}", tf.keras.applications.EfficientNetB0)
    
    base_model = backbone_cls(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    base_model.trainable = trainable_base

    inputs = layers.Input(shape=input_shape, name="image_input")
    
    # Feature extraction
    features = base_model(inputs, training=False)
    
    # Classification head
    x = layers.GlobalAveragePooling2D(name="avg_pool")(features)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.5, name="dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="ai_probability_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"ImageGuard_EfficientNet{variant}")

    # Compile with binary cross-entropy and tracking metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return model
