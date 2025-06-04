"""Train and evaluate product classification model."""

import logging
from tensorflow.keras import layers, models

logger = logging.getLogger(__name__)


def train_model(train_data, val_data):
    """Train CNN model for product classification.

    Args:
        train_data: Training dataset
        val_data: Validation dataset

    Returns:
        Trained model
    """
    try:
        model = models.Sequential(
            [
                layers.Rescaling(1.0 / 255, input_shape=(224, 224, 3)),
                layers.Conv2D(32, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dense(train_data.class_names.__len__(), activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        model.fit(train_data, validation_data=val_data, epochs=5)
        logger.info("Model training completed")
        return model
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise
