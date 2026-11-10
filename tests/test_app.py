import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'app.py'


class AppTests(unittest.TestCase):
    def start(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertFalse(app.exception)
        return app

    def test_empty_review(self):
        app = self.start()
        app.button[0].click().run()
        self.assertTrue(app.warning)
        self.assertFalse(app.metric)
        self.assertFalse(app.exception)

    def test_real_prediction_and_sample_change(self):
        app = self.start()
        app.selectbox[0].select('Loved it').run()
        self.assertIn('Excellent', app.text_area[0].value)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 1)
        self.assertIn('Positive sentiment', app.success[0].value)
        app.selectbox[0].select('Disappointed').run()
        self.assertFalse(app.metric)

    def test_negative_result(self):
        with patch('predict_sentiment.predict_sentiment', return_value={'sentiment': 'Negative', 'confidence': 0.87}):
            app = self.start()
            app.text_area[0].set_value('Not good.').run()
            app.button[0].click().run()
            self.assertEqual(app.metric[0].value, '87.0%')
            self.assertEqual(app.error[0].value, 'Negative sentiment')

    def test_errors_are_handled(self):
        for error in [FileNotFoundError('missing'), RuntimeError('invalid artifact')]:
            with self.subTest(error=error), patch('predict_sentiment.predict_sentiment', side_effect=error):
                app = self.start()
                app.text_area[0].set_value('A review').run()
                app.button[0].click().run()
                self.assertTrue(app.error)
                self.assertFalse(app.exception)
                self.assertFalse(app.metric)


if __name__ == '__main__':
    unittest.main()
