"""Train baseline classifiers using only the existing Phase 3 feature split."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB

from prepare_features import DEFAULT_OUTPUT, ROOT, load_features


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average='binary', pos_label=1, zero_division=0,
    )
    return {
        'accuracy': float(accuracy_score(y_test, predictions)),
        'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
    }, confusion_matrix(y_test, predictions, labels=[0, 1])


def train_and_evaluate(features):
    """Fit on X_train/y_train only; use the unchanged test set for evaluation."""
    models = {
        'MultinomialNB': MultinomialNB(),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    }
    rows, matrices = [], {}
    for name, model in models.items():
        model.fit(features['X_train'], features['y_train'])
        metrics, matrices[name] = evaluate_model(model, features['X_test'], features['y_test'])
        rows.append({'model': name, **metrics})
    table = pd.DataFrame(rows).sort_values(['f1', 'accuracy', 'model'], ascending=[False, False, True])
    return models, table, matrices


def artifact_hashes(directory):
    result = {}
    for name in ['X_train.npz', 'X_test.npz', 'split.npz', 'vectorizer.joblib', 'metadata.json']:
        with (directory / name).open('rb') as stream:
            result[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return result


def run(feature_dir=DEFAULT_OUTPUT, output=ROOT / 'artifacts' / 'models'):
    feature_dir, output = Path(feature_dir), Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing results: {output}')
    before = artifact_hashes(feature_dir)
    features = load_features(feature_dir)
    assert not np.intersect1d(features['train_indices'], features['test_indices']).size
    for split in ['train', 'test']:
        assert features[f'X_{split}'].shape[0] == len(features[f'y_{split}'])
    models, table, matrices = train_and_evaluate(features)
    output.mkdir(parents=True, exist_ok=False)
    table.to_csv(output / 'results.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, (name, matrix) in zip(axes, matrices.items()):
        ConfusionMatrixDisplay(matrix, display_labels=['Negative (0)', 'Positive (1)']).plot(
            ax=ax, cmap='Blues', colorbar=False, values_format='d',
        )
        ax.set_title(name)
    fig.savefig(output / 'confusion_matrices.png', dpi=160)
    plt.close(fig)
    best_name = table.iloc[0]['model']
    joblib.dump(models[best_name], output / 'best_model.joblib')
    restored = joblib.load(output / 'best_model.joblib')
    np.testing.assert_array_equal(restored.predict(features['X_test']), models[best_name].predict(features['X_test']))
    assert artifact_hashes(feature_dir) == before, 'Phase 3 artifacts changed'
    metadata = {
        'best_model': best_name, 'selection_metric': 'test binary F1 (positive class 1)',
        'tie_break': 'accuracy, then model name',
        'evaluation_note': 'Best of these two on this test set; selection uses test scores, so these are not an independent final estimate for the selected model.',
        'feature_artifacts': str(feature_dir.resolve()), 'feature_sha256': before,
        'confusion_matrices': {name: matrix.tolist() for name, matrix in matrices.items()},
        'confusion_matrix_order': 'rows=true, columns=predicted; labels=[0,1]',
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(table.to_string(index=False, float_format=lambda value: f'{value:.4f}'))
    print(f'Saved best model: {best_name}; Phase 3 artifacts unchanged; reload predictions verified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts' / 'models')
    args = parser.parse_args()
    run(args.features, args.output)
