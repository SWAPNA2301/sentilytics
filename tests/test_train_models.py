import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

from train_models import evaluate_model, train_and_evaluate


class ModelTests(unittest.TestCase):
    def test_metrics_and_confusion_matrix(self):
        class FixedModel:
            def predict(self, X):
                return np.array([0, 1, 1, 1])
        metrics, matrix = evaluate_model(FixedModel(), None, [0, 0, 1, 1])
        self.assertEqual(metrics['accuracy'], 0.75)
        self.assertAlmostEqual(metrics['precision'], 2 / 3)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertAlmostEqual(metrics['f1'], 0.8)
        np.testing.assert_array_equal(matrix, [[1, 1], [0, 2]])

    def test_fit_receives_training_data_only(self):
        features = dict(X_train=csr_matrix([[2, 0], [0, 2]]), y_train=np.array([0, 1]),
                        X_test=csr_matrix([[1, 0], [0, 1]]), y_test=np.array([0, 1]))
        with patch('train_models.MultinomialNB') as nb, patch('train_models.LogisticRegression') as lr:
            for constructor in [nb, lr]:
                constructor.return_value.predict.return_value = np.array([0, 1])
            _, table, _ = train_and_evaluate(features)
            for constructor in [nb, lr]:
                args = constructor.return_value.fit.call_args.args
                self.assertIs(args[0], features['X_train'])
                self.assertIs(args[1], features['y_train'])
                self.assertEqual(constructor.return_value.fit.call_count, 1)
            self.assertEqual(len(table), 2)
            lr.assert_called_once_with(max_iter=1000, random_state=42)


if __name__ == '__main__':
    unittest.main()
