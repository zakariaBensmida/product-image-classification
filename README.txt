```markdown
# Multiclass Product Image Classification

A Python project for classifying product images (e.g., electronics, clothing) using deep learning, with a Streamlit dashboard.

## Features
- Image preprocessing with TensorFlow
- CNN model training with Keras
- Interactive Streamlit dashboard for predictions
- Linting with black and ruff
- Environment management with python-dotenv
- Optional AWS S3 integration via aws_utils 

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Download dataset from [E-commerce Product Images](https://www.kaggle.com/datasets/kumararunited22/ecommerce-product-images) and place in `data/raw/`.
3. Train model: `python main.py`
4. View dashboard: `streamlit run src/app.py`

## Notes
- Global Python used; virtual environments recommended for production.
- Sensitive data (e.g., `.env`, `data/`) excluded via `.gitignore`.
- AWS placeholders in `.env.example` demonstrate cloud compatibility (not used by default).

## Project Structure
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── output/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model.py
│   ├── app.py
│   └── aws_utils.py
├── tests/
├── .env.example
├── .gitignore
├── main.py
├── pyproject.toml
├── requirements.txt
└── README.md
