import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Define paths relative to this file
# Assuming structure:
# root/
#   2022/data/raw/
#   2025/src/dataset.py
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "2022" / "data" / "raw"


def load_isot_dataset(test_size=0.2, seed=42):
    """
    Loads the ISOT Fake News Dataset from the 2022 folder.
    Returns a Hugging Face DatasetDict with 'train' and 'test' splits.
    """
    fake_path = DATA_DIR / "Fake1.csv"
    true_path = DATA_DIR / "True1.csv"

    if not fake_path.exists() or not true_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset files at {DATA_DIR}. Please ensure 2022/data/raw exists."
        )

    print(f"Loading data from {DATA_DIR}...")
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Add labels: 0 for Fake, 1 for True (Standard convention)
    fake_df["label"] = 0
    true_df["label"] = 1

    # Combine dataframes
    df = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)

    # Combine title and text for better context
    # Handling missing values just in case
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["content"] = df["title"] + " " + df["text"]

    # Keep only necessary columns
    df = df[["content", "label"]]

    # Split into train and test
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )

    # Convert to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

    return DatasetDict({"train": train_dataset, "test": test_dataset})


if __name__ == "__main__":
    # Test the loading
    try:
        ds = load_isot_dataset()
        print("Dataset loaded successfully!")
        print(ds)
        print("Sample entry:", ds["train"][0])
    except Exception as e:
        print(f"Error loading dataset: {e}")
