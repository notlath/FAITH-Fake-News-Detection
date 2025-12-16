from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from pathlib import Path
import torch
import sys
import os
from dotenv import load_dotenv
from firecrawl import FirecrawlApp

# Load environment variables
load_dotenv()

# Add 2022 src to path to import legacy modules
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR / "2022" / "src"))

try:
    from Process_pipeline import FakeNewsDetector
except ImportError as e:
    FakeNewsDetector = None
    print(f"Warning: Could not import FakeNewsDetector from 2022/src. Error: {e}")
    import traceback

    traceback.print_exc()

app = FastAPI(
    title="Fake News Detector API",
    description="API for detecting fake news using 2022 (ML) and 2025 (Transformer) models.",
    version="2.0.0",
)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths
BASE_DIR_2025 = Path(__file__).resolve().parent.parent
MODEL_PATH_2025 = BASE_DIR_2025 / "models" / "trained_model"

MODEL_PATH_2022 = ROOT_DIR / "2022" / "models" / "trained_model"
VECTORIZER_PATH_2022 = ROOT_DIR / "2022" / "vectorizers" / "Vectorizer"

# Global variables
classifier_2025 = None
detector_2022 = None


@app.on_event("startup")
async def load_models():
    global classifier_2025, detector_2022

    # Load 2025 Model
    print(f"Loading 2025 model from {MODEL_PATH_2025}...")
    try:
        if not MODEL_PATH_2025.exists():
            print(f"Warning: 2025 Model not found at {MODEL_PATH_2025}. Using default.")
            model_name = "distilroberta-base"
        else:
            model_name = str(MODEL_PATH_2025)

        device = 0 if torch.cuda.is_available() else -1
        classifier_2025 = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=device,
            truncation=True,
            max_length=512,
        )
        print("2025 Model loaded successfully!")
    except Exception as e:
        print(f"Error loading 2025 model: {e}")

    # Load 2022 Model
    print(f"Loading 2022 model from {MODEL_PATH_2022}...")
    print(f"Vectorizer path: {VECTORIZER_PATH_2022}")

    if FakeNewsDetector is None:
        print("Cannot load 2022 model because FakeNewsDetector class failed to import.")
    else:
        try:
            if MODEL_PATH_2022.exists() and VECTORIZER_PATH_2022.exists():
                detector_2022 = FakeNewsDetector(MODEL_PATH_2022, VECTORIZER_PATH_2022)
                print("2022 Model loaded successfully!")
            else:
                print(
                    f"Skipping 2022 model load. Model exists: {MODEL_PATH_2022.exists()}, Vectorizer exists: {VECTORIZER_PATH_2022.exists()}"
                )
        except Exception as e:
            print(f"Error loading 2022 model: {e}")
            import traceback

            traceback.print_exc()


class NewsRequest(BaseModel):
    text: str
    title: str = ""


class ScrapeRequest(BaseModel):
    url: str
    api_key: str = ""


class NewsResponse(BaseModel):
    label: str
    confidence: float
    verdict: str
    model: str


@app.get("/")
def root():
    return {"message": "Fake News Detector API (2022 & 2025) is running."}


@app.post("/predict/2025", response_model=NewsResponse)
def predict_2025(request: NewsRequest):
    if not classifier_2025:
        raise HTTPException(status_code=503, detail="2025 Model not loaded.")

    content = f"{request.title} {request.text}".strip()
    if not content:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        prediction = classifier_2025(content)
        result = prediction[0]
        label = result["label"]
        score = result["score"]

        verdict = "Unknown"
        if label in ["REAL", "LABEL_1"]:
            verdict = "Reliable"
        elif label in ["FAKE", "LABEL_0"]:
            verdict = "Fake"

        return {
            "label": label,
            "confidence": score,
            "verdict": verdict,
            "model": "2025 (Transformer)",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/2022", response_model=NewsResponse)
def predict_2022(request: NewsRequest):
    if not detector_2022:
        raise HTTPException(status_code=503, detail="2022 Model not loaded.")

    content = f"{request.title} {request.text}".strip()
    if not content:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        # The 2022 model returns a string label directly
        verdict_text = detector_2022.predict(content)

        # Map back to structure
        # Verdict is "Unreliable (Fake)" or "Reliable (True)"
        label = "FAKE" if "Fake" in verdict_text else "REAL"
        verdict = "Fake" if "Fake" in verdict_text else "Reliable"

        return {
            "label": label,
            "confidence": 1.0,
            "verdict": verdict,
            "model": "2022 (Classic ML)",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/scrape")
def scrape_url(request: ScrapeRequest):
    api_key = request.api_key or os.getenv("FIRECRAWL_API_KEY")

    if not api_key:
        print("Error: No Firecrawl API Key provided.")
        raise HTTPException(status_code=400, detail="Firecrawl API Key required.")

    print(
        f"Using Firecrawl API Key: {api_key[:4]}...{api_key[-4:] if len(api_key) > 8 else ''}"
    )

    try:
        firecrawl = FirecrawlApp(api_key=api_key)
        scrape_result = firecrawl.scrape(request.url, formats=["markdown"])

        markdown = ""
        if isinstance(scrape_result, dict):
            markdown = scrape_result.get("markdown", "")
        elif hasattr(scrape_result, "markdown"):
            markdown = scrape_result.markdown

        if not markdown:
            raise HTTPException(status_code=404, detail="No content found.")

        return {"content": markdown}
    except Exception as e:
        error_msg = str(e)
        print(f"Scraping error: {error_msg}")
        if "Unauthorized" in error_msg:
            raise HTTPException(
                status_code=401,
                detail=f"Firecrawl Unauthorized: Invalid API Key. {error_msg}",
            )
        raise HTTPException(status_code=500, detail=f"Scraping error: {error_msg}")
