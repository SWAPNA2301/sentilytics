import unittest

import numpy as np

from prepare_features import build_features


class FeatureTests(unittest.TestCase):
    def test_stratification_reproducibility_and_training_only_vocabulary(self):
        texts = [f'common token unique{i}' for i in range(100)]
        labels = [0, 1] * 50
        first = build_features(texts, labels)
        second = build_features(texts, labels)
        self.assertEqual(first['X_train'].shape[0], 80)
        self.assertEqual(first['X_test'].shape[0], 20)
        np.testing.assert_array_equal(np.bincount(first['y_train']), [40, 40])
        np.testing.assert_array_equal(np.bincount(first['y_test']), [10, 10])
        np.testing.assert_array_equal(first['train_indices'], second['train_indices'])
        np.testing.assert_array_equal(first['test_indices'], second['test_indices'])
        self.assertFalse(set(first['train_indices']) & set(first['test_indices']))
        vocabulary = first['vectorizer'].vocabulary_
        for index in first['test_indices']:
            self.assertNotIn(f'unique{index}', vocabulary)
        for index in first['train_indices']:
            self.assertIn(f'unique{index}', vocabulary)
        self.assertIn('common token', vocabulary)
        self.assertLessEqual(len(vocabulary), 5000)
        self.assertEqual(first['X_train'].shape[1], first['X_test'].shape[1])


if __name__ == '__main__':
    unittest.main()
