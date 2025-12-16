# Fake News Detection Project

## 2022 Archive

The original implementation using classical machine learning (Logistic Regression, Naive Bayes) has been archived in the `2022/` directory.

To run the archived version:

1. Navigate to the directory: `cd 2022`
2. Install requirements: `pip install -r requirements.txt`
3. Run the pipeline: `python Process_pipeline.py`

## Modern Implementation (2025)

The project has been modernized to use a Transformer-based model (DistilRoBERTa) served via a FastAPI backend, with a Next.js frontend.

### Backend & Model (`2025/`)

The backend handles data processing, model training, and inference.

- **Model**: DistilRoBERTa fine-tuned on the ISOT dataset.
- **API**: FastAPI.
- **Location**: [2025/](2025/)

To get started with the backend:

1. Navigate to `2025/`.
2. Follow the instructions in [2025/README.md](2025/README.md).

### Frontend (`web-app/`)

A modern web interface built with Next.js to interact with the detection model.

- **Framework**: Next.js (React).
- **Location**: [web-app/](web-app/)

To get started with the frontend:

1. Navigate to `web-app/`.
2. Follow the instructions in [web-app/README.md](web-app/README.md).
