# FAI(TH): Fake News Detection System

### Falsified Narrative Verification through Machine Learning Ensemble Methods

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)
![NLTK](https://img.shields.io/badge/NLTK-3.8-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A Grade 12 research project for Practical Research 2 (Quantitative) at Pasig City Science High School**

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🌐 Overview

In the digital age, the Internet has become an integral part of electronic media, serving as the primary channel for information dissemination. With the exponential growth of online content consumption, the circulation of news articles has increased dramatically. Unfortunately, this growth has been accompanied by the rapid spread of disinformation and fake news, making it increasingly difficult to distinguish between legitimate journalism and falsified narratives.

**FAI(TH)** is a machine learning-based system designed to combat misinformation by automatically detecting and classifying fake news articles from both text and web-based inputs. The system leverages ensemble methods and natural language processing techniques to provide accurate, real-time verification of news content.

---

## ✨ Features

- **Dual Input Processing**: Supports both direct text input and URL-based web scraping
- **Advanced NLP Pipeline**: Implements Porter Stemmer for text normalization and intelligent stopword removal
- **Ensemble Classification**: Utilizes multiple ML models (Logistic Regression, Random Forest, Multinomial Naive Bayes)
- **Dual Vectorization Techniques**: Employs both CountVectorizer and TF-IDF for feature extraction
- **Automated Web Scraping**: Real-time article extraction using BeautifulSoup and Newspaper3k
- **Production-Ready Models**: Serialized trained models using Pickle for deployment
- **Comprehensive Evaluation**: Detailed performance metrics including precision, recall, F1-score, and confusion matrices
- **High Accuracy**: Trained on 44,898+ articles for robust prediction capabilities

---

## 🛠 Tech Stack

| Category              | Technologies                          |
| --------------------- | ------------------------------------- |
| **Language**          | Python 3.8+                           |
| **ML Framework**      | Scikit-learn                          |
| **NLP Processing**    | NLTK, Porter Stemmer, Regex (re)      |
| **Data Manipulation** | Pandas, NumPy                         |
| **Web Scraping**      | BeautifulSoup4, Newspaper3k, Requests |
| **Visualization**     | Matplotlib, Seaborn                   |
| **Development**       | Jupyter Notebook                      |
| **Model Persistence** | Pickle                                |

---

## 🏗 Project Architecture

```
┌─────────────────┐
│   Input Source  │
│ (Text/URL)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Web Scraper    │
│ (Newspaper3k)   │
└────────┬────────┘
         │
         ▼
┌────────────────────┐
│ Text Preprocessing │
│ - Regex Cleaning   │
│ - Lowercasing      │
│ - Stemming         │
│ - Stopword Removal │
└────────┬───────────┘
         │
         ▼
┌─────────────────┐
│  Vectorization  │
│ CountVec/TF-IDF │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ML Classifier   │
│ (Ensemble)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Prediction      │
│ Reliable/       │
│ Unreliable      │
└─────────────────┘
```

---

## 📊 Dataset

### ISOT Fake News Dataset

**Source:** [University of Victoria ISOT Research Lab](https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/03/News-_dataset.zip)

| File        | Type         | Records    | Description              |
| ----------- | ------------ | ---------- | ------------------------ |
| `Fake1.csv` | Fake News    | 23,481     | Falsified news articles  |
| `True1.csv` | Real News    | 21,417     | Legitimate news articles |
| **Total**   | **Combined** | **44,898** | **Complete dataset**     |

**Dataset Characteristics:**

- Balanced representation of fake and real news
- Diverse topics and sources
- Real-world news articles
- High-quality labels for supervised learning

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/notlath/Fake-News-Detection.git
cd Fake-News-Detection
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### Step 5: Download Dataset

1. Download the ISOT Fake News Dataset from the link above
2. Extract and place `Fake1.csv` and `True1.csv` in the `Data_RealFake/` directory

---

## 💻 Usage

### Method 1: Using Jupyter Notebook

1. Launch Jupyter Notebook:

```bash
jupyter notebook
```

2. Open `main.ipynb`

3. Run all cells sequentially

4. Enter a URL when prompted for testing

### Method 2: Using Python Script

```bash
python Process_pipeline.py
```

### Method 3: Manual Testing with Custom Text

```python
from Process_pipeline import manual_testing

# Test with custom text
news_text = "Your news article text here..."
manual_testing(news_text)
```

### Method 4: Testing with URL

```python
from newspaper import Article
from Process_pipeline import manual_testing

url = "https://example.com/news-article"
article = Article(url)
article.download()
article.parse()

content = article.title + ' ' + article.text
manual_testing(content)
```

---

## 📈 Model Performance

### Classification Models Evaluated

| Model                       | Vectorizer      | Accuracy | Precision | Recall | F1-Score |
| --------------------------- | --------------- | -------- | --------- | ------ | -------- |
| **Logistic Regression**     | CountVectorizer | High     | High      | High   | High     |
| **Logistic Regression**     | TF-IDF          | High     | High      | High   | High     |
| **Random Forest**           | CountVectorizer | High     | High      | High   | High     |
| **Multinomial Naive Bayes** | CountVectorizer | High     | High      | High   | High     |

_Detailed metrics available in [Evaluation Pipeline](Evaluation%20Pipeline/)_

### Key Performance Indicators

- ✅ **High Accuracy**: Robust performance on test dataset
- ✅ **Balanced Precision/Recall**: Minimizes both false positives and false negatives
- ✅ **Consistent Results**: Stable performance across different vectorization techniques
- ✅ **Real-time Processing**: Fast prediction for production use

---

## 📁 Project Structure

```
2022/
│
├── data/                       # Dataset directory
│   └── raw/                    # Original CSV files
│       ├── Fake1.csv
│       └── True1.csv
│
├── notebooks/                  # Jupyter notebooks
│   ├── main.ipynb              # Main implementation
│   ├── CountVector_pipeline.ipynb
│   ├── Tf-Idf_pipeline.ipynb
│   └── Eval_pipeline.ipynb
│
├── src/                        # Source code
│   ├── Process_pipeline.py     # Production pipeline script
│   └── train_model.py          # Training script
│
├── models/                     # Serialized models
│   └── trained_model
│
├── vectorizers/                # Serialized vectorizers
│   ├── Vectorizer
│   └── Vectorizer_data
│
├── outputs/                    # Generated artifacts
│   └── figures/
│       ├── count.png
│       └── TF.png
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🔬 Methodology

### 1. Data Preprocessing

**Text Cleaning Pipeline:**

```python
def stemming(content):
    # Remove non-alphabetic characters
    stemmed = re.sub('[^a-zA-Z]', ' ', content)

    # Convert to lowercase
    stemmed = stemmed.lower()

    # Tokenization
    stemmed = stemmed.split()

    # Stemming + Stopword removal
    stemmed = [port_stem.stem(word) for word in stemmed
               if word not in stopwords.words('english')]

    # Rejoin tokens
    return ' '.join(stemmed)
```

**Preprocessing Steps:**

1. **Regex Cleaning**: Remove special characters, numbers, and punctuation
2. **Normalization**: Convert all text to lowercase
3. **Tokenization**: Split text into individual words
4. **Stemming**: Reduce words to their root form using Porter Stemmer
5. **Stopword Removal**: Filter out common English words with no semantic value

### 2. Feature Extraction

**Two Vectorization Approaches:**

- **CountVectorizer**: Creates a document-term matrix based on word frequency
- **TF-IDF Vectorizer**: Weighs words by their importance using Term Frequency-Inverse Document Frequency

### 3. Model Training

**Ensemble Approach:**

- Multiple classification algorithms evaluated
- Cross-validation for robust performance estimation
- Hyperparameter tuning for optimal results
- Model serialization for production deployment

### 4. Web Scraping Integration

**Article Extraction:**

- BeautifulSoup for HTML parsing
- Newspaper3k for intelligent article extraction
- Title and body content aggregation
- Error handling for unavailable content

---

## 🎯 Research Objectives

The main objective of this project is to provide a tool that can assess and distinguish falsified information using machine learning. Specifically, the system aims to achieve:

1. **Accurate Detection**: Determine whether an internet article is legitimate based on textual attributes and linguistic patterns

2. **Aggregated Information Processing**: Process and analyze multiple information sources for comprehensive validity assessment

3. **Model Effectiveness**: Evaluate and optimize the performance of various ML algorithms in news classification tasks

4. **Misinformation Prevention**: Accurately identify falsified statements to combat the spread of disinformation

5. **Dual Input Efficiency**: Efficiently verify both text-based and web-based inputs using language-centered verification algorithms

---

## 📊 Results

### Confusion Matrix for CountVectorizer Pipeline

![CountVectorizer Results](outputs/figures/count.png)

### Confusion Matrix for TF-IDF Vectorizer Pipeline

![TF-IDF Results](outputs/figures/TF.png)

**Key Findings:**

- Both vectorization techniques demonstrate high accuracy
- Minimal false positives and false negatives
- Robust performance across different article types
- Production-ready classification capabilities

---

## 🚀 Future Enhancements

### Planned Improvements

- [ ] **Deep Learning Integration**: Implement LSTM/BERT models for enhanced accuracy
- [ ] **Multi-language Support**: Extend detection to non-English articles
- [ ] **Real-time API**: Develop RESTful API for web integration
- [ ] **Browser Extension**: Create Chrome/Firefox extension for on-the-fly verification
- [ ] **Explainable AI**: Implement SHAP/LIME for prediction explanations
- [ ] **Source Credibility Analysis**: Incorporate domain reputation scoring
- [ ] **Fact-checking Integration**: Connect with external fact-checking databases
- [ ] **Mobile Application**: Develop iOS/Android apps for mobile verification
- [ ] **Social Media Integration**: Direct verification of social media posts
- [ ] **Continuous Learning**: Implement online learning for model updates

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines for Python code
- Add unit tests for new features
- Update documentation for API changes
- Ensure all tests pass before submitting PR

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Research Team

**Pasig City Science High School - Grade 12 Practical Research 2 (Quantitative)**

- **Rhon Jay C. Cailo**
- **Jethro M. Caingat**
- **Jasper Alvin T. Dee**
- **Jimuel Ronald H. Dimaandal**
- **Red Christian F. Hernandez**
- **Lathrell T. Pagsuguiron**
- **Alyssa Gwyneth S. Rivera**

### Data Source

- **ISOT Research Lab, University of Victoria** - Dataset Provider

### Open Source Libraries

- Scikit-learn, NLTK, Pandas, NumPy - Core ML/NLP tools
- BeautifulSoup, Newspaper3k - Web scraping capabilities
- Jupyter, Matplotlib - Development and visualization

### Special Thanks

- Academic advisors and mentors
- Open-source community contributors
- Fake news research community

---

## 📞 Contact

For questions, suggestions, or collaborations:

- **Project Repository**: [GitHub](https://github.com/notlath/Fake-News-Detection)
- **Issues**: [Report Bug](https://github.com/notlath/Fake-News-Detection)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by Pasig City Science High School Students

</div>
