import pandas as pd
import pickle
from pathlib import Path
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk

# Ensure resources are downloaded
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Define Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
MODEL_PATH = BASE_DIR / "models" / "trained_model"
VECTORIZER_PATH = BASE_DIR / "vectorizers" / "Vectorizer"


def load_data():
    print("Loading data...")
    fake_df = pd.read_csv(DATA_DIR / "Fake1.csv")
    true_df = pd.read_csv(DATA_DIR / "True1.csv")

    # Add labels
    fake_df["label"] = 0
    true_df["label"] = 1

    # Combine
    df = pd.concat([fake_df, true_df], axis=0)

    # We only need text and label.
    # Note: The original dataset usually has 'title' and 'text'.
    # We'll combine them for better accuracy, or just use 'text' depending on the original logic.
    # Let's check columns first, but usually it's 'title', 'text', 'subject', 'date'.
    # For safety, let's just use 'text' + 'title' if available, or just 'text'.

    # Handling potential missing values
    df = df.fillna("")

    # Combine title and text for training
    df["content"] = df["title"] + " " + df["text"]

    return df


def stemming(content):
    port_stem = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    stemmed = re.sub("[^a-zA-Z]", " ", content)
    stemmed = stemmed.lower()
    stemmed = stemmed.split()
    stemmed = [port_stem.stem(word) for word in stemmed if not word in stop_words]
    stemmed = " ".join(stemmed)
    return stemmed


def train():
    df = load_data()

    print("Preprocessing data (this may take a while)...")
    # Apply stemming
    # For speed in this demonstration, we might want to sample if the dataset is huge,
    # but let's try full dataset first.
    df["content"] = df["content"].apply(stemming)

    X = df["content"].values
    y = df["label"].values

    print("Vectorizing...")
    # Using TfidfVectorizer as it's generally better than CountVectorizer,
    # but the original error mentioned CountVectorizer.
    # However, the user's code used `vectorizer.transform`.
    # Let's stick to TfidfVectorizer as it's standard for this dataset (ISOT/Fake News Dataset).
    vectorizer = TfidfVectorizer()
    vectorizer.fit(X)
    X_transformed = vectorizer.transform(X)

    print("Training Model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.2, stratify=y, random_state=2
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluation
    train_acc = accuracy_score(model.predict(X_train), y_train)
    test_acc = accuracy_score(model.predict(X_test), y_test)
    print(f"Training Accuracy: {train_acc}")
    print(f"Test Accuracy: {test_acc}")

    print("Saving model and vectorizer...")
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    with open(VECTORIZER_PATH, "wb") as file:
        pickle.dump(vectorizer, file)

    print("Done! You can now run Process_pipeline.py")


if __name__ == "__main__":
    train()
