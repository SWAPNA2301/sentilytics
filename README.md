<<<<<<< HEAD
# sentilytics
=======
# Sentilytics

A local sentiment analysis app for English product reviews. Enter a review or choose an example to see **Positive / Negative** sentiment and the model's predicted-class probability.

**GitHub description:** Product review sentiment analysis with TF-IDF, scikit-learn and Streamlit; reproducible preprocessing, model evaluation and local predictions.

**Tech stack:** Python, pandas, NumPy, SciPy sparse matrices, scikit-learn, joblib, Matplotlib, Streamlit and standard-library unittest.

## Quick start

Use Python 3.14 (the development runtime was Python 3.14.2). From the project directory:

First clone your published repository and enter its root. Install dependencies below, then restore the two trusted artifacts or complete **Dataset and offline reproduction** before launching the app. Direct dependencies are pinned to the locally validated versions; transitive dependencies are not locked. A new environment installation has not been independently verified.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On macOS/Linux, activate with `source .venv/bin/activate`. Open the local URL printed by Streamlit, normally `http://localhost:8501`.

### Required local artifacts

The app requires this matching pair from Phases 3 and 4:

```text
artifacts/tfidf/vectorizer.joblib
artifacts/models/best_model.joblib
```

Datasets and generated artifacts are deliberately excluded from Git. A fresh clone does not contain them. Restore trusted artifacts from your local project or run the offline preparation workflow below. The app itself never trains a model, creates a split, or fits TF-IDF. Joblib files execute Python during loading: load only trusted files. Restore the model and vectorizer together, using compatible dependency versions. The saved artifacts were generated with scikit-learn 1.9.1, joblib 1.6.0, NumPy 2.5.4 and SciPy 1.18.1; dependency ranges alone do not guarantee cross-version serialization compatibility.

## Usage

1. Choose a sample or enter a custom review.
2. Click **Predict sentiment**.
3. Read the predicted class and its probability. Changing the input clears the previous result.

Empty reviews receive a prompt; unavailable or invalid artifacts produce an error message. Model probability is not a calibrated guarantee. The app does not save submitted reviews.

Command-line prediction:

```powershell
python predict_sentiment.py "Excellent food! I love it."
```

Reusable Python API:

```python
from predict_sentiment import predict_sentiment

result = predict_sentiment("Not good. I would never buy it again.")
print(result["sentiment"], result["confidence"])
```

The API returns a dictionary with `sentiment` and a `confidence` probability from 0 to 1. Empty or missing text raises `ValueError`.

## Methodology

1. **Preparation:** inspect source columns, missing values, duplicates and scores. Map scores 1–2 to 0 (Negative), 4–5 to 1 (Positive), exclude 3, and retain the first eligible row for each exact review text. Sample 25,000 reviews per class with seed 42.
2. **Cleaning:** lowercase; remove HTML tags, URLs, unnecessary punctuation and extra whitespace. Preserve negation, expand common negative contractions, and handle missing text safely. Keep original fields and append `clean_text`.
3. **Features:** stratified 80/20 split with seed 42: 40,000 training and 10,000 test reviews, evenly balanced by class. Fit `TfidfVectorizer(max_features=5000, ngram_range=(1, 2))` on training text only; transform both splits. Save sparse matrices, aligned labels, row indices and vectorizer.
4. **Models:** train `MultinomialNB` and `LogisticRegression(max_iter=1000, random_state=42)` only on the saved training matrix. Compare on the existing test matrix and select the highest positive-class F1.
5. **Inference:** reuse `clean_text()`, the saved vectorizer and logistic regression model. Transform only; never refit or retrain.

## Measured model results

Results from the existing 10,000-review test set, recorded in the local `artifacts/models/results.csv`:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| **Logistic Regression** | **0.8964** | **0.9006** | **0.8912** | **0.8959** |
| Multinomial Naive Bayes | 0.8791 | 0.8842 | 0.8724 | 0.8783 |

Precision, recall and F1 use Positive (1). Confusion matrices, in `[[TN, FP], [FN, TP]]` order:

- Logistic Regression: `[[4508, 492], [544, 4456]]`
- Multinomial Naive Bayes: `[[4429, 571], [638, 4362]]`

The best model was selected using test F1, so these scores are comparison results, not an independent final estimate after selection. No test data was used to fit TF-IDF or classifier parameters. Neutral reviews are excluded; mixed sentiment, sarcasm, unfamiliar vocabulary and out-of-domain text may be misclassified. Exact-text deduplication does not eliminate all near-duplicate reviews or product/user overlap across splits.

## Dataset and offline reproduction

Download **[Amazon Fine Food Reviews by SNAP on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)** (568,454 reviews; the dataset page lists CC0). Sign in to Kaggle if requested, click Download, and extract `Reviews.csv` from the archive into a new `data/` directory in the project root. Neither `database.sqlite` nor a Kaggle API credential is needed. Do not put the archive or credentials into Git.

The dataset cites J. McAuley and J. Leskovec, *From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews*, WWW 2013. Follow the dataset's attribution guidance when publishing research.

Place the original reviews CSV at `data/Reviews.csv`, including `Text` and `Score` columns. Run these steps only when intentionally preparing/training offline:

```powershell
python prepare_reviews.py
python clean_reviews.py
python prepare_features.py
python train_models.py
```

The first step reads the original CSV and writes `data/processed_reviews.csv` and EDA reports. It can replace its generated output on reruns. Subsequent steps refuse to overwrite existing cleaned data or artifact directories; use their `--output` option for a fresh location when needed. The defaults form the pipeline used by the app. The original source CSV is never overwritten.

After those four commands, run `python -m streamlit run app.py`. This regenerates the matching trained model and vectorizer locally; the reported historical metrics above remain the recorded results, and rerun results may vary with dataset or environment changes.

### Obtain a trained model without training

No public pretrained download is currently published. Request the trusted matching `best_model.joblib` and `vectorizer.joblib` from the project owner, or copy them from an existing local run into the paths shown above. Ask for `artifacts/models/metadata.json` as well: its vectorizer SHA-256 links the classifier to its feature vocabulary. For the full integration test suite, also restore the complete Phase 3 artifact directory, including the saved matrices and labels. Alternatively, regenerate everything using the offline workflow. A future GitHub Release may distribute these files separately; this repository does not claim one exists.

## Tests

With the trusted Phase 3/4 artifacts installed:

```powershell
python -m unittest discover -p "test_*.py"
```

Tests cover cleaning, split isolation, metrics, train-only fitting, artifact loading, prediction consistency and Streamlit interactions. App smoke tests exercise empty input, samples, a real saved-model prediction, negative results and error handling. They do not retrain the production models or overwrite artifacts.

## Project structure

```text
sentilytics/
├── app.py                    # Streamlit entry point
├── predict_sentiment.py      # Inference API and CLI
├── clean_reviews.py          # Shared text preprocessing
├── prepare_reviews.py        # Offline data preparation
├── prepare_features.py       # Offline TF-IDF preparation
├── train_models.py           # Offline model comparison
├── tests/                    # Unit and app smoke tests
├── .streamlit/config.toml    # Public UI theme (no secrets)
├── requirements.txt
├── README.md
├── data/                     # Local only; ignored
├── artifacts/                # Local only; ignored
└── reports/                  # Local only; ignored
```

The small reusable modules stay at the root so existing imports, CLI commands and Streamlit launches remain compatible. Run commands from this directory.

| File | Purpose |
|---|---|
| [app.py](app.py) | Responsive Streamlit interface |
| [predict_sentiment.py](predict_sentiment.py) | Cached artifact loading, API and CLI |
| [clean_reviews.py](clean_reviews.py) | Reusable text cleaning |
| [prepare_reviews.py](prepare_reviews.py) | Source inspection, sampling and EDA |
| [prepare_features.py](prepare_features.py) | Offline TF-IDF preparation and artifact loading |
| [train_models.py](train_models.py) | Offline model fitting and evaluation |

The Git ignore rules exclude datasets, virtual environments, secrets, generated reports, feature matrices and serialized models. Keep these local or distribute them separately through a trusted channel.
>>>>>>> 081af0a (Initial commit: Sentilytics NLP sentiment classifier)
