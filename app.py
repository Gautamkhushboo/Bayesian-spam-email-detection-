import os
import io
import csv
import json
import re
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK stopwords available
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

app = Flask(__name__)

# Paths
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
STATIC_FOLDER = os.path.join(PROJECT_ROOT, 'static')
TEMPLATE_FOLDER = os.path.join(PROJECT_ROOT, 'templates')

# Load trained model and vectorizer
clf = joblib.load(os.path.join(MODEL_DIR, 'spam_classifier.pkl'))
vectorizer = joblib.load(os.path.join(MODEL_DIR, 'vectorizer.pkl'))

# Load metrics for dashboard
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, 'r') as f:
        metrics = json.load(f)
else:
    metrics = {}

# Compute feature names and class log-probabilities
feature_names = vectorizer.get_feature_names_out()
vocab_dict = vectorizer.vocabulary_

# clf.classes_ : ['ham', 'spam']
ham_idx = np.where(clf.classes_ == 'ham')[0][0]
spam_idx = np.where(clf.classes_ == 'spam')[0][0]

log_prob_ham = clf.feature_log_prob_[ham_idx]
log_prob_spam = clf.feature_log_prob_[spam_idx]
log_odds = log_prob_spam - log_prob_ham  # > 0 means spam-leaning, < 0 means ham-leaning

# Compute top vocabulary keywords for dashboard
top_spam_indices = np.argsort(log_odds)[-20:][::-1]
top_ham_indices = np.argsort(log_odds)[:20]

top_spam_keywords = [{"word": feature_names[i], "score": round(float(log_odds[i]), 2)} for i in top_spam_indices]
top_ham_keywords = [{"word": feature_names[i], "score": round(float(abs(log_odds[i])), 2)} for i in top_ham_indices]


def preprocess_text(text: str, enable_stemming: bool = True) -> str:
    """Standard text preprocessing matching training pipeline."""
    if not isinstance(text, str):
        return ""
    text_lower = text.lower()
    clean_text = ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in text_lower)
    tokens = clean_text.split()
    tokens = [t for t in tokens if t not in stop_words]
    if enable_stemming:
        tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)


def analyze_words(raw_text: str):
    """Tokenizes text and computes token-level Bayesian log-odds explainability."""
    # Split raw text into words and punctuation while preserving original token tokens
    tokens_raw = re.findall(r"[\w']+|[.,!?;:\-–—()\"/$%#@&*+<=>[\]^{|}]", raw_text)
    token_details = []
    word_impact_map = {}

    for raw_token in tokens_raw:
        lower_token = raw_token.lower()
        # check if it's alphanumeric word
        if re.match(r"^[a-zA-Z0-9]+$", lower_token):
            is_stop = lower_token in stop_words
            stemmed = stemmer.stem(lower_token) if not is_stop else lower_token
            in_vocab = stemmed in vocab_dict and not is_stop
            
            if in_vocab:
                feat_idx = vocab_dict[stemmed]
                odds = float(log_odds[feat_idx])
                
                if odds > 0.35:
                    impact = 'spam'
                elif odds < -0.35:
                    impact = 'ham'
                else:
                    impact = 'neutral'
                    
                token_info = {
                    "raw": raw_token,
                    "stem": stemmed,
                    "impact": impact,
                    "score": round(odds, 3),
                    "is_stop": False,
                    "in_vocab": True
                }
                
                if stemmed not in word_impact_map:
                    word_impact_map[stemmed] = {
                        "word": stemmed,
                        "display": raw_token,
                        "count": 1,
                        "impact": impact,
                        "score": round(odds, 3)
                    }
                else:
                    word_impact_map[stemmed]["count"] += 1
            else:
                token_info = {
                    "raw": raw_token,
                    "stem": stemmed,
                    "impact": "neutral",
                    "score": 0.0,
                    "is_stop": is_stop,
                    "in_vocab": False
                }
        else:
            token_info = {
                "raw": raw_token,
                "stem": raw_token,
                "impact": "neutral",
                "score": 0.0,
                "is_stop": False,
                "in_vocab": False
            }
        token_details.append(token_info)

    # Sort word impacts by magnitude of influence
    impact_list = list(word_impact_map.values())
    impact_list.sort(key=lambda x: abs(x["score"]), reverse=True)

    return token_details, impact_list


@app.route('/')
def index():
    """Render the single-page dashboard."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to classify a single message with probability and explainability."""
    if not request.is_json:
        return jsonify({"error": "Invalid JSON payload"}), 400
    
    data = request.get_json()
    message = data.get('message', '').strip()
    if not message:
        return jsonify({"error": "Message cannot be empty"}), 400

    # Preprocess text
    processed_msg = preprocess_text(message)
    X_vec = vectorizer.transform([processed_msg])
    
    # Model prediction and probabilities
    pred = clf.predict(X_vec)[0]
    probs = clf.predict_proba(X_vec)[0]
    prob_dict = dict(zip(clf.classes_, probs))
    
    spam_prob = float(prob_dict.get('spam', 0.0))
    ham_prob = float(prob_dict.get('ham', 0.0))
    confidence = max(spam_prob, ham_prob)

    # Word-level explainability analysis
    token_details, impact_list = analyze_words(message)

    # Prior probabilities in training
    prior_ham = float(np.exp(clf.class_log_prior_[ham_idx]))
    prior_spam = float(np.exp(clf.class_log_prior_[spam_idx]))

    response = {
        "prediction": pred,
        "spam_probability": round(spam_prob, 4),
        "ham_probability": round(ham_prob, 4),
        "confidence": round(confidence, 4),
        "token_details": token_details,
        "word_impacts": impact_list,
        "bayes_math": {
            "prior_spam": round(prior_spam, 4),
            "prior_ham": round(prior_ham, 4),
            "tokens_matched": len(impact_list),
            "processed_text": processed_msg
        }
    }
    return jsonify(response)


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Endpoint for bulk message classification via JSON or uploaded CSV/TXT."""
    messages = []
    
    if request.is_json:
        data = request.get_json()
        messages = data.get('messages', [])
    elif 'file' in request.files:
        file = request.files['file']
        filename = file.filename.lower()
        
        content = file.read().decode('utf-8', errors='ignore')
        
        if filename.endswith('.csv'):
            reader = csv.reader(io.StringIO(content))
            header = None
            for row in reader:
                if not row:
                    continue
                if header is None:
                    # check if row looks like header
                    header = [c.strip().lower() for c in row]
                    msg_col = -1
                    for idx, col in enumerate(header):
                        if col in ['message', 'text', 'email', 'body', 'content', 'sms']:
                            msg_col = idx
                            break
                    if msg_col == -1:
                        # not a recognised header row; assume first column is message
                        msg_col = 0
                        messages.append(row[0].strip())
                    continue
                else:
                    if len(row) > msg_col and row[msg_col].strip():
                        messages.append(row[msg_col].strip())
        else:
            # Assume line by line text file
            for line in content.splitlines():
                if line.strip():
                    messages.append(line.strip())

    if not messages:
        return jsonify({"error": "No valid messages found to classify"}), 400

    # Limit batch size to prevent server timeout
    max_batch = 500
    if len(messages) > max_batch:
        messages = messages[:max_batch]

    processed_list = [preprocess_text(m) for m in messages]
    X_batch = vectorizer.transform(processed_list)
    preds = clf.predict(X_batch)
    probs = clf.predict_proba(X_batch)

    results = []
    spam_count = 0
    ham_count = 0
    total_conf = 0.0

    for i, (msg, pred, prob) in enumerate(zip(messages, preds, probs)):
        prob_dict = dict(zip(clf.classes_, prob))
        s_prob = float(prob_dict.get('spam', 0.0))
        h_prob = float(prob_dict.get('ham', 0.0))
        conf = max(s_prob, h_prob)
        total_conf += conf
        
        if pred == 'spam':
            spam_count += 1
        else:
            ham_count += 1
            
        results.append({
            "id": i + 1,
            "message": (msg[:90] + '...') if len(msg) > 90 else msg,
            "full_message": msg,
            "prediction": pred,
            "spam_probability": round(s_prob, 4),
            "ham_probability": round(h_prob, 4),
            "confidence": round(conf, 4)
        })

    summary = {
        "total": len(messages),
        "spam_count": spam_count,
        "ham_count": ham_count,
        "spam_rate": round((spam_count / len(messages)) * 100, 1) if messages else 0,
        "avg_confidence": round(total_conf / len(messages), 4) if messages else 0,
        "results": results
    }

    return jsonify(summary)


@app.route('/stats', methods=['GET'])
def stats():
    """Return model performance metrics and top predictive keywords."""
    stats_data = dict(metrics)
    stats_data['vocabulary_size'] = len(feature_names)
    stats_data['top_spam_keywords'] = top_spam_keywords
    stats_data['top_ham_keywords'] = top_ham_keywords
    return jsonify(stats_data)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model_loaded": clf is not None})


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
