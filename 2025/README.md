# Fake News Detection (2025 Modern Implementation)

This directory contains the modernized implementation of the Fake News Detection system, utilizing Transformer-based models (DistilRoBERTa) and a FastAPI backend.

## Project Structure

- `src/`: Source code for data loading and model training.
  - `dataset.py`: Loads and prepares the ISOT dataset.
  - `train.py`: Fine-tunes the Transformer model.
- `app/`: Application code.
  - `main.py`: FastAPI backend for serving predictions.
- `models/`: Directory where trained models are saved.
- `notebooks/`: Jupyter notebooks for experimentation.

## Setup

1.  **Create a virtual environment** (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Training the Model

To fine-tune the model on the ISOT dataset (located in `../2022/data/raw`):

```bash
cd src
python train.py
```

This will:

1.  Load the data.
2.  Tokenize it using the `distilroberta-base` tokenizer.
3.  Fine-tune the model for 3 epochs.
4.  Save the best model to `../models/trained_model`.

## Running the API

Once the model is trained (or to use the base model for testing), start the API:

```bash
cd app
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

### API Endpoints

- `GET /`: Health check.
- `POST /predict`: Predict if a news article is fake or real.

**Example Request:**

```json
{
  "title": "Breaking News",
  "text": "Scientists discover water on the sun."
}
```

## Key Improvements over 2022 Version

- **Model**: Moved from Logistic Regression/Naive Bayes to **DistilRoBERTa** (Transformer).
- **Preprocessing**: Removed manual stemming and stopword removal in favor of **Subword Tokenization**.
- **Context**: The model now understands the context and order of words, not just their frequency.
- **Architecture**: Separated training logic from the inference API.
