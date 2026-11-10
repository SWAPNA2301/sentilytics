"""Prepare TF-IDF features; no classifier fitting. Run before Phase 4."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / 'artifacts' / 'tfidf'


def build_features(texts, labels):
    """Split by row position, fit on train only, and return reusable artifacts."""
    texts = pd.Series(texts).fillna('').astype(str).reset_index(drop=True)
    labels = np.asarray(labels)
    if len(texts) != len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError('Expected equally sized texts and binary sentiment labels')
    train_indices, test_indices = train_test_split(
        np.arange(len(texts)), test_size=0.2, stratify=labels, random_state=42,
    )
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(texts.iloc[train_indices])
    X_test = vectorizer.transform(texts.iloc[test_indices])
    return dict(vectorizer=vectorizer, X_train=X_train, X_test=X_test,
                y_train=labels[train_indices], y_test=labels[test_indices],
                train_indices=train_indices, test_indices=test_indices)


def load_features(directory=DEFAULT_OUTPUT):
    """Load locally generated, trusted artifacts for Phase 4 model training.

    joblib files must not be loaded from untrusted sources.
    """
    directory = Path(directory)
    result = {name: load_npz(directory / f'{name}.npz') for name in ['X_train', 'X_test']}
    with np.load(directory / 'split.npz', allow_pickle=False) as split:
        result.update({name: split[name] for name in split.files})
    result['vectorizer'] = joblib.load(directory / 'vectorizer.joblib')
    return result


def prepare_features(source, output=DEFAULT_OUTPUT):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite artifacts: {output}')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    frame = pd.read_csv(source, usecols=['clean_text', 'sentiment'])
    result = build_features(frame.clean_text, frame.sentiment)
    output.mkdir(parents=True, exist_ok=False)
    for name in ['X_train', 'X_test']:
        save_npz(output / f'{name}.npz', result[name])
    np.savez_compressed(output / 'split.npz', **{
        name: result[name] for name in ['y_train', 'y_test', 'train_indices', 'test_indices']
    })
    joblib.dump(result['vectorizer'], output / 'vectorizer.joblib')
    metadata = {
        'source': str(source), 'source_sha256': source_hash,
        'seed': 42, 'test_size': 0.2, 'text_column': 'clean_text',
        'label_column': 'sentiment', 'fit_scope': 'training rows only',
        'max_features': 5000, 'ngram_range': [1, 2],
        'sklearn_version': sklearn.__version__,
        'train_shape': list(result['X_train'].shape),
        'test_shape': list(result['X_test'].shape),
        'vocabulary_size': len(result['vectorizer'].vocabulary_),
        'train_class_counts': pd.Series(result['y_train']).value_counts().sort_index().to_dict(),
        'test_class_counts': pd.Series(result['y_test']).value_counts().sort_index().to_dict(),
    }
    restored = load_features(output)
    for name in ['X_train', 'X_test']:
        assert (result[name] != restored[name]).nnz == 0
    for name in ['y_train', 'y_test', 'train_indices', 'test_indices']:
        np.testing.assert_array_equal(result[name], restored[name])
    assert not np.intersect1d(result['train_indices'], result['test_indices']).size
    assert len(result['train_indices']) + len(result['test_indices']) == len(frame)
    transformed = restored['vectorizer'].transform(frame.clean_text.fillna('').iloc[result['test_indices']])
    assert (transformed != result['X_test']).nnz == 0
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data' / 'cleaned_reviews.csv')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare_features(args.input, args.output)
