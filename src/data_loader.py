"""Load and preprocess product image data."""

import logging
from pathlib import Path
import tensorflow as tf

logger = logging.getLogger(__name__)


def load_data(data_dir: Path):
    """Load and preprocess product images.

    Args:
        data_dir: Path to raw data directory

    Returns:
        Train and validation datasets
    """
    try:
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=42,
            image_size=(224, 224),
            batch_size=32,
            label_mode="categorical",
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=(224, 224),
            batch_size=32,
            label_mode="categorical",
        )
        logger.info("Data loaded successfully")
        return train_ds, val_ds
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        raise
