import pickle
import re
import os
from pathlib import Path
from typing import Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from firecrawl import FirecrawlApp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define paths using Pathlib
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "trained_model"
VECTORIZER_PATH = BASE_DIR / "vectorizers" / "Vectorizer"


class FakeNewsDetector:
    def __init__(self, model_path: Path, vectorizer_path: Path):
        self.port_stem = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords")
            self.stop_words = set(stopwords.words("english"))
        self.model = self._load_pickle(model_path)
        self.vectorizer = self._load_pickle(vectorizer_path)

    @staticmethod
    def _load_pickle(path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Resource not found at {path}")
        with open(path, "rb") as file:
            return pickle.load(file)

    def preprocess_text(self, content: str) -> str:
        """Cleans and stems text to match training data format."""
        # Remove non-alphabetic characters (This also removes Markdown symbols from Firecrawl)
        stemmed = re.sub("[^a-zA-Z]", " ", content)
        stemmed = stemmed.lower()
        stemmed = stemmed.split()
        # Apply stemming and remove stopwords
        stemmed = [
            self.port_stem.stem(word) for word in stemmed if word not in self.stop_words
        ]
        return " ".join(stemmed)

    def predict(self, text: str) -> str:
        """Runs the prediction pipeline."""
        processed_text = self.preprocess_text(text)

        # Transform input using the loaded vectorizer
        vectorized_input = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(vectorized_input)

        return self._get_label(prediction[0])

    @staticmethod
    def _get_label(n: int) -> str:
        if n == 0:
            return "Unreliable (Fake)"
        elif n == 1:
            return "Reliable (True)"
        return "Unknown"


class NewsScraper:
    def __init__(self, api_key: str):
        self.app = FirecrawlApp(api_key=api_key)

    def fetch_article(self, url: str) -> Optional[str]:
        """Fetches article text using Firecrawl to bypass bot detection."""
        print("  > Contacting Firecrawl...")
        try:
            # Scrape the URL for markdown content
            scrape_result = self.app.scrape(url, formats=["markdown"])

            # Handle both dictionary and object responses
            if isinstance(scrape_result, dict):
                return scrape_result.get("markdown")
            elif hasattr(scrape_result, "markdown"):
                return scrape_result.markdown

            print("  > Firecrawl did not return markdown content.")
            return None

        except Exception as e:
            print(f"  > Firecrawl Error: {e}")
            return None


def main():
    # Initialize system
    try:
        detector = FakeNewsDetector(MODEL_PATH, VECTORIZER_PATH)
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        return

    # Get API Key
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        print("\nTo use Firecrawl, you need an API key from https://firecrawl.dev")
        api_key = input("Enter your Firecrawl API Key: ").strip()
        if not api_key:
            print("API Key is required.")
            return

    scraper = NewsScraper(api_key)

    # Get Input
    url = input("\nEnter News URL: ").strip()
    if not url:
        print("No URL provided.")
        return

    # Process
    print("Fetching article...")
    article_text = scraper.fetch_article(url)

    if article_text:
        print(
            f"\n--- Scraped Text Preview ---\n{article_text[:500]}...\n----------------------------\n"
        )
        print("Analyzing...")
        result = detector.predict(article_text)
        print(f"\nResult: {result}")
    else:
        print("Could not extract valid text from the URL.")


if __name__ == "__main__":
    main()
