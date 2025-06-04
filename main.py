"""Main script for product image classification."""

import logging
from pathlib import Path
from src.data_loader import load_data
from src.model import train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the product classification pipeline."""
    logger.info("Starting product image classification")
    data_dir = Path("data/raw")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    try:
        train_data, val_data = load_data(data_dir)
        model = train_model(train_data, val_data)
        model.save(output_dir / "model.h5")
        logger.info("Model trained and saved")
    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()
