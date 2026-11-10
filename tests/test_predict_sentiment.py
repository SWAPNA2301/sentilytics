import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from clean_reviews import clean_text
from predict_sentiment import load_artifacts, predict_sentiment


class PredictionTests(unittest.TestCase):
    def tearDown(self):
        load_artifacts.cache_clear()

    def test_artifact_loading(self):
        model, vectorizer = load_artifacts()
        self.assertEqual(set(model.classes_), {0, 1})
        self.assertEqual(model.n_features_in_, len(vectorizer.vocabulary_))
        self.assertIs(load_artifacts()[0], model)

    def test_prediction_matches_saved_model_without_refitting(self):
        model, vectorizer = load_artifacts()
        for text in ['<b>Excellent</b> food! I love it.', 'Terrible taste. Never buying again.']:
            with self.subTest(text=text):
                expected = model.predict_proba(vectorizer.transform([clean_text(text)]))[0]
                with patch.object(model, 'fit', side_effect=AssertionError('Must not train')), \
                     patch.object(vectorizer, 'fit', side_effect=AssertionError('Must not fit')), \
                     patch.object(vectorizer, 'fit_transform', side_effect=AssertionError('Must not fit')):
                    result = predict_sentiment(text)
                label = model.classes_[expected.argmax()]
                self.assertEqual(result['sentiment'], 'Positive' if label == 1 else 'Negative')
                self.assertAlmostEqual(result['confidence'], float(expected.max()))
                self.assertTrue(0 <= result['confidence'] <= 1)

    def test_empty_input_does_not_load_artifacts(self):
        with patch('predict_sentiment.load_artifacts') as loader:
            for text in [None, '', '   ', '<br>!!!', 'https://example.com']:
                with self.subTest(text=text), self.assertRaises(ValueError):
                    predict_sentiment(text)
            loader.assert_not_called()

    def test_missing_artifact(self):
        load_artifacts.cache_clear()
        with tempfile.TemporaryDirectory() as directory:
            with patch('predict_sentiment.MODEL_PATH', Path(directory) / 'missing.joblib'):
                with self.assertRaisesRegex(FileNotFoundError, 'artifact is missing'):
                    load_artifacts()


if __name__ == '__main__':
    unittest.main()
