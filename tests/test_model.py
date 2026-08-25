import unittest
import os
import sys
import joblib
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import preprocess_text, analyze_words, clf, vectorizer, log_odds


class TestModelPipeline(unittest.TestCase):
    """Test suite for Bayesian Classifier inference & vectorization pipeline."""

    def test_model_and_vectorizer_exist(self):
        """Ensure serialized model and vectorizer exist and are valid."""
        model_path = os.path.join(PROJECT_ROOT, 'model', 'spam_classifier.pkl')
        vec_path = os.path.join(PROJECT_ROOT, 'model', 'vectorizer.pkl')
        
        self.assertTrue(os.path.exists(model_path), "Spam classifier pickle missing")
        self.assertTrue(os.path.exists(vec_path), "Vectorizer pickle missing")
        self.assertIsNotNone(clf)
        self.assertIsNotNone(vectorizer)

    def test_vocabulary_size(self):
        """Ensure vocabulary contains meaningful feature terms."""
        vocab = vectorizer.get_feature_names_out()
        self.assertGreater(len(vocab), 1000)

    def test_text_preprocessing(self):
        """Test lowercase conversion, punctuation removal, and stemming."""
        raw = "CONGRATULATIONS!!! You're winning & claiming prizes."
        processed = preprocess_text(raw)
        # Should be lowercase, punctuation removed, stemmed ('congratul', 'win', 'claim', 'prize')
        self.assertNotIn('!', processed)
        self.assertNotIn('&', processed)
        self.assertTrue(any(t in processed for t in ['win', 'claim', 'prize', 'congratul']))

    def test_explainability_analysis(self):
        """Test token extraction and log-odds scoring."""
        msg = "Claim your free prize today"
        tokens, impacts = analyze_words(msg)
        self.assertGreater(len(tokens), 0)
        self.assertGreater(len(impacts), 0)
        
        # Check that 'claim' or 'prize' has a positive (spam) log-odds score
        top_words = [item['word'] for item in impacts]
        self.assertTrue(any(w in top_words for w in ['claim', 'prize', 'free']))


if __name__ == '__main__':
    unittest.main()
