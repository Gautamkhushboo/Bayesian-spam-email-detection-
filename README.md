# Bayesian Spam Email Detection

## Overview

This project implements a simple **email spam filter** using **Bayesian classification** (Multinomial Naïve Bayes). It demonstrates how to:

1. Load a labeled dataset of email messages (spam vs. ham).
2. Convert raw text into a numeric feature matrix with `CountVectorizer`.
3. Train a Naïve Bayes classifier.
4. Predict whether a new email is spam.

The implementation is deliberately lightweight and can be used as a starting point for more sophisticated pipelines (e.g., adding preprocessing, using TF‑IDF, or incorporating more data).

## Project Structure

```
BayesianSpamDetection/
├─ data/                # Example dataset (CSV) – you can replace with your own
│   └─ spam.csv         # Minimal sample (spam, ham) – optional
├─ src/                
│   └─ spam_classifier.py   # Core implementation
├─ requirements.txt    # Python dependencies
└─ README.md           # This file
```

## Installation

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/Scripts/activate  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python src/spam_classifier.py --train data/spam.csv
```

The script will:
- Train the model on the supplied CSV (columns: `label`, `message`).
- Save the trained model to `model.pkl`.
- Provide a simple interactive prompt to classify new messages.

You can also import the `SpamClassifier` class in your own Python code:

```python
from src.spam_classifier import SpamClassifier

clf = SpamClassifier.load('model.pkl')
print(clf.predict('Congratulations! You won a free iPhone!'))
```

## License

MIT License – feel free to adapt and extend.
