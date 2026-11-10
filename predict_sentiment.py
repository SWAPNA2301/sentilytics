"""Predict with trusted local Phase 3/4 artifacts; never fit or train."""

import argparse
from functools import lru_cache
from pathlib import Path

import joblib

from clean_reviews import clean_text

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'artifacts/models/best_model.joblib'
VECTORIZER_PATH = ROOT / 'artifacts/tfidf/vectorizer.joblib'


@lru_cache(maxsize=1)
def load_artifacts():
    """Load and cache the trusted local model and fitted vectorizer."""
    for path in [MODEL_PATH, VECTORIZER_PATH]:
        if not path.is_file():
            raise FileNotFoundError(f'Required prediction artifact is missing: {path}')
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    if set(model.classes_) != {0, 1}:
        raise ValueError('Expected model classes 0 and 1')
    if model.n_features_in_ != len(vectorizer.vocabulary_):
        raise ValueError('Model and vectorizer feature counts do not match')
    return model, vectorizer


def predict_sentiment(text):
    """Return sentiment and predicted-class probability (0–1).

    Confidence is the model's probability, not a calibrated guarantee.
    Empty or missing text raises ValueError before loading artifacts.
    """
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError('Enter a review containing text after cleaning.')
    model, vectorizer = load_artifacts()
    features = vectorizer.transform([cleaned])
    probabilities = model.predict_proba(features)[0]
    best = int(probabilities.argmax())
    return {
        'sentiment': 'Positive' if model.classes_[best] == 1 else 'Negative',
        'confidence': float(probabilities[best]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text', nargs='?', help='Review text; prompts if omitted')
    args = parser.parse_args()
    try:
        text = args.text if args.text is not None else input('Review: ')
        result = predict_sentiment(text)
    except (ValueError, FileNotFoundError, EOFError) as error:
        parser.exit(2, f'Error: {error}\n')
    print(f"{result['sentiment']} | confidence: {result['confidence']:.2%}")


if __name__ == '__main__':
    main()
