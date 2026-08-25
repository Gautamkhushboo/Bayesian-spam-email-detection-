import unittest
import json
import io
import os
import sys

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import app


class TestSpamDetectionAPI(unittest.TestCase):
    """Test suite for Bayesian Spam Detection Flask REST API."""

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index_page(self):
        """Test GET / renders HTML dashboard."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Bayesian Spam Guard', response.data)
        self.assertIn(b'Live Email Inspector', response.data)

    def test_health_endpoint(self):
        """Test GET /health returns status healthy."""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data.get('status'), 'healthy')
        self.assertTrue(data.get('model_loaded'))

    def test_stats_endpoint(self):
        """Test GET /stats returns model metrics and top keywords."""
        response = self.app.get('/stats')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('accuracy', data)
        self.assertIn('precision', data)
        self.assertIn('recall', data)
        self.assertIn('f1_score', data)
        self.assertIn('vocabulary_size', data)
        self.assertIn('top_spam_keywords', data)
        self.assertIn('top_ham_keywords', data)
        self.assertGreater(data['accuracy'], 0.90)
        self.assertGreater(len(data['top_spam_keywords']), 0)

    def test_predict_spam_message(self):
        """Test POST /predict classifies obvious spam correctly."""
        payload = {
            "message": "CONGRATULATIONS! You have won £1,000 cash prize! Call 09061701461 to claim your reward instantly."
        }
        response = self.app.post('/predict',
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['prediction'], 'spam')
        self.assertGreater(data['spam_probability'], 0.8)
        self.assertIn('token_details', data)
        self.assertIn('word_impacts', data)
        self.assertGreater(len(data['word_impacts']), 0)

    def test_predict_ham_message(self):
        """Test POST /predict classifies genuine message correctly."""
        payload = {
            "message": "Hey, let's meet up at the library tomorrow at 3pm to work on our science project."
        }
        response = self.app.post('/predict',
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['prediction'], 'ham')
        self.assertGreater(data['ham_probability'], 0.7)

    def test_predict_empty_message(self):
        """Test POST /predict with empty message returns 400 error."""
        response = self.app.post('/predict',
                                 data=json.dumps({"message": "   "}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_predict_invalid_payload(self):
        """Test POST /predict without JSON returns 400."""
        response = self.app.post('/predict', data="not json")
        self.assertEqual(response.status_code, 400)

    def test_predict_special_characters_and_emojis(self):
        """Test POST /predict gracefully handles unicode and emojis."""
        payload = {"message": "🔥 Claim $500 free gift card now! 🎁 $$$ 🚀"}
        response = self.app.post('/predict',
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('prediction', data)

    def test_batch_predict_json(self):
        """Test POST /batch-predict with JSON message list."""
        messages = [
            "WINNER! Claim your free £500 gift voucher now by calling 0800123456.",
            "Can you please review the attached slides before our team call?",
            "Urgent: Your account password will expire in 2 hours."
        ]
        response = self.app.post('/batch-predict',
                                 data=json.dumps({"messages": messages}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['total'], 3)
        self.assertEqual(len(data['results']), 3)
        self.assertIn('spam_count', data)
        self.assertIn('ham_count', data)
        self.assertIn('avg_confidence', data)

    def test_batch_predict_csv_upload(self):
        """Test POST /batch-predict with CSV file multipart upload."""
        csv_data = "message\n" \
                   "\"URGENT: Win a free holiday to Hawaii! Text WIN to 77222\"\n" \
                   "\"See you at lunch today around noon.\"\n"
        
        file_obj = (io.BytesIO(csv_data.encode('utf-8')), 'test_batch.csv')
        response = self.app.post('/batch-predict',
                                 data={'file': file_obj},
                                 content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['total'], 2)
        self.assertEqual(len(data['results']), 2)


if __name__ == '__main__':
    unittest.main()
