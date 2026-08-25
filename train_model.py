import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib
import json

# ------------------------------------------------------------
# 1. Ensure NLTK resources are available
# ------------------------------------------------------------
nltk.download('stopwords', quiet=True)

# ------------------------------------------------------------
# 2. Paths
# ------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DATASET_PATH = os.path.join(DATASET_DIR, 'spam.csv')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')

# ------------------------------------------------------------
# 3. Download dataset if not present (SMSSpamCollection from UCI)
# ------------------------------------------------------------
if not os.path.isfile(DATASET_PATH):
    print('Downloading dataset...')
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    zip_path = os.path.join(DATASET_DIR, 'smsspamcollection.zip')
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATASET_DIR)
    # The extracted file is named SMSSpamCollection
    extracted_path = os.path.join(DATASET_DIR, 'SMSSpamCollection')
    # Convert to CSV with columns ['label', 'message']
    df_raw = pd.read_csv(extracted_path, sep='\t', header=None, names=['label', 'message'])
    df_raw.to_csv(DATASET_PATH, index=False)
    os.remove(zip_path)
    os.remove(extracted_path)
    print('Dataset saved to', DATASET_PATH)
else:
    print('Dataset already exists at', DATASET_PATH)

# ------------------------------------------------------------
# 4. Load dataset
# ------------------------------------------------------------
df = pd.read_csv(DATASET_PATH)
# Normalize labels
df['label'] = df['label'].map({'spam': 'spam', 'ham': 'ham'})

# ------------------------------------------------------------
# 5. Text preprocessing utilities
# ------------------------------------------------------------
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text, enable_stemming=True):
    # Lowercase
    text = text.lower()
    # Remove punctuation (keep only alphanumerics and spaces)
    text = ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in text)
    # Tokenize
    tokens = text.split()
    # Remove stopwords
    tokens = [t for t in tokens if t not in stop_words]
    # Stemming (optional)
    if enable_stemming:
        tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

print('Preprocessing texts...')
X_processed = df['message'].apply(lambda x: preprocess(x))
y = df['label']

# ------------------------------------------------------------
# 6. Split data
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------
# 7. Feature extraction (CountVectorizer – can switch to TFIDF by changing class)
# ------------------------------------------------------------
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ------------------------------------------------------------
# 8. Train Multinomial Naive Bayes
# ------------------------------------------------------------
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# ------------------------------------------------------------
# 9. Evaluate
# ------------------------------------------------------------
y_pred = clf.predict(X_test_vec)
probs = clf.predict_proba(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='spam')
recall = recall_score(y_test, y_pred, pos_label='spam')
f1 = f1_score(y_test, y_pred, pos_label='spam')
conf_mat = confusion_matrix(y_test, y_pred, labels=['spam', 'ham']).tolist()
report = classification_report(y_test, y_pred, output_dict=True)

print('\nModel Evaluation:')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision (spam): {precision:.4f}')
print(f'Recall (spam): {recall:.4f}')
print(f'F1 Score (spam): {f1:.4f}')
print('Confusion Matrix:', conf_mat)

# ------------------------------------------------------------
# 10. Save model, vectorizer, and metrics
# ------------------------------------------------------------
joblib.dump(clf, os.path.join(MODEL_DIR, 'spam_classifier.pkl'))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'vectorizer.pkl'))

metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'confusion_matrix': conf_mat,
    'classification_report': report,
    'total_records': len(df),
    'spam_count': int(df[df['label'] == 'spam'].shape[0]),
    'ham_count': int(df[df['label'] == 'ham'].shape[0]),
}
with open(METRICS_PATH, 'w') as f:
    json.dump(metrics, f, indent=2)

print('\nModel, vectorizer, and metrics saved to', MODEL_DIR)
