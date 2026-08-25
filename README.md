# Bayesian Spam Email Detection & Explainability Platform

A full-stack, production-ready machine learning application and interactive analytics dashboard that detects and explains **SPAM** vs **HAM** emails using a **Multinomial Naive Bayes** classifier with token-level log-odds mathematical attribution.

---

## 🌟 Key Features

- **⚡ Real-Time Bayesian Inference**: High-speed text vectorization and probability classification with confidence estimation.
- **🔍 Token-Level Explainability Engine**: Calculates mathematical log-odds ratios ($\Delta(w) = \ln P(w|\text{Spam}) - \ln P(w|\text{Ham})$) to highlight specific spam and ham trigger words with hover tooltips and attribution strength bars.
- **📁 Batch CSV / TXT Classifier**: Drag-and-drop file upload, multi-line paste, aggregate batch statistics, and 1-click **Export Predictions CSV**.
- **🎨 Glassmorphism 2.0 Dashboard**: Modern dark theme with ambient glowing accents, smooth transitions, and responsive tabbed architecture (*Live Inspector*, *Batch Classifier*, *Model Analytics*, and *Bayes Math*).
- **📊 Interactive Chart.js Visualizations**:
  - Class Distribution Donut Chart (5,572 records)
  - Confusion Matrix Performance Breakdown (True Spam, True Ham, False Positive/Negative)
  - Top Predictive Keywords Bar Chart (ranked by discriminative power)
- **🧠 Educational Bayes' Theorem Guide**: Visual step-by-step breakdown of priors, likelihoods with Laplace smoothing ($\alpha = 1.0$), and log-space computation.
- **🧪 Comprehensive Test Suite**: 14 automated unit and integration tests covering REST endpoints, inference correctness, and edge cases.

---

## 🏗️ System Architecture

```
+-------------------------------------------------------------+
|                      Browser Frontend                       |
|   HTML5 + Bootstrap 5 + Vanilla CSS (Glassmorphic) + JS     |
|   Chart.js Dynamic Visualizations & Explainability UI       |
+------------------------------+------------------------------+
                               |  HTTP REST (JSON / Multipart)
                               v
+-------------------------------------------------------------+
|                     Flask REST API (app.py)                 |
|   /predict          -> Single classification & token weights|
|   /batch-predict    -> Multi-line / CSV batch processor     |
|   /stats            -> Metrics & top vocabulary keywords    |
|   /health           -> Service health check                 |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|             Machine Learning Core (Scikit-Learn)            |
|   • CountVectorizer (Bag of Words, Stopwords, Stemming)     |
|   • Multinomial Naive Bayes (Laplace Smoothing α=1.0)       |
|   • Serialized Artifacts: spam_classifier.pkl, metrics.json|
+-------------------------------------------------------------+
```

---

## 📈 Model Performance Metrics

Trained on the **UCI SMS Spam Collection** (5,572 messages: 4,825 Ham, 747 Spam):

| Metric | Score | Description |
|---|---|---|
| **Accuracy** | **98.48%** | Overall correct classifications on unseen test set |
| **Spam Precision** | **96.48%** | Ultra-low false positive rate (legitimate emails won't be blocked) |
| **Spam Recall** | **91.95%** | Percentage of spam successfully detected |
| **F1-Score** | **94.16%** | Harmonic mean of precision and recall |
| **Vocabulary** | **6,401 terms** | N-gram token features extracted |

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Gautamkhushboo/Bayesian-spam-email-detection-.git
cd Bayesian-spam-email-detection-
```

### 2. Set Up Virtual Environment (Optional but recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train Model (Optional — Pre-trained model included)
```bash
python train_model.py
```

### 5. Launch the Web Application
```bash
python app.py
```
Open your browser at `http://127.0.0.1:5000/`.

---

## 🧪 Running Automated Tests

Run the full test suite with Python's built-in `unittest`:
```bash
python -m unittest discover tests
```

---

## 📡 API Specification

### 1. Classify Single Message
- **Endpoint**: `POST /predict`
- **Payload**:
  ```json
  {
    "message": "CONGRATULATIONS! You have won a $1,000 gift card! Claim now."
  }
  ```
- **Response**:
  ```json
  {
    "prediction": "spam",
    "spam_probability": 0.9982,
    "ham_probability": 0.0018,
    "confidence": 0.9982,
    "word_impacts": [
      { "word": "claim", "impact": "spam", "score": 5.34, "count": 1 },
      { "word": "prize", "impact": "spam", "score": 5.00, "count": 1 }
    ],
    "token_details": [...]
  }
  ```

### 2. Batch Classification
- **Endpoint**: `POST /batch-predict`
- **Accepts**: JSON `{"messages": ["msg1", "msg2", ...]}` or multipart CSV/TXT file upload with column `message`.
- **Returns**: Aggregate counts, spam rate %, average confidence, and per-row prediction details.

### 3. Model Statistics & Keywords
- **Endpoint**: `GET /stats`
- **Returns**: Accuracy, precision, recall, confusion matrix, and top 20 spam/ham vocabulary words with log-odds scores.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, Flask, Joblib, NumPy, Pandas
- **Machine Learning & NLP**: Scikit-Learn, NLTK (Stopwords, PorterStemmer)
- **Frontend**: HTML5, Vanilla CSS (Glassmorphism 2.0), JavaScript (ES6+), Bootstrap 5, Chart.js, FontAwesome 6
- **Testing**: Python `unittest`

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
