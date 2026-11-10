"""Inspect, deduplicate, sample and plot reviews; never modify the source."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "Reviews.csv"
OUTPUT = ROOT / "data" / "processed_reviews.csv"
REPORTS = ROOT / "reports"
SEED = 42
PER_CLASS = 25_000


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    source_hash = digest(SOURCE)
    df = pd.read_csv(SOURCE)
    if not {"Score", "Text"}.issubset(df.columns):
        raise ValueError("Source must contain Score and Text columns")
    report = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": source_hash,
        "rows": len(df),
        "columns": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "nulls": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_review_texts": int(df.loc[df.Text.notna(), "Text"].duplicated().sum()),
        "rating_distribution": df.Score.value_counts(dropna=False).sort_index().to_dict(),
        "seed": SEED,
        "requested_per_class": PER_CLASS,
    }
    scores = pd.to_numeric(df.Score, errors="coerce")
    valid_text = df.Text.notna() & df.Text.str.strip().ne("")
    report["missing_or_blank_texts"] = int((~valid_text).sum())
    report["neutral_rows_excluded"] = int(scores.eq(3).sum())
    report["invalid_or_missing_scores"] = int((~scores.isin([1, 2, 3, 4, 5])).sum())
    clean = df.loc[valid_text & scores.isin([1, 2, 4, 5])].copy()
    clean["sentiment"] = scores.loc[clean.index].ge(4).astype(int)
    report["conflicting_texts"] = int((clean.groupby("Text").sentiment.nunique() > 1).sum())
    before = len(clean)
    clean = clean.drop_duplicates(subset="Text", keep="first")
    report["duplicate_texts_removed_after_filtering"] = before - len(clean)
    report["deduplication_policy"] = "Exact Text match; retain first eligible source row, including conflicting labels."
    available = clean.sentiment.value_counts().reindex([0, 1], fill_value=0)
    report["available_per_class"] = available.to_dict()
    sampled = pd.concat([
        clean.loc[clean.sentiment.eq(label)].sample(n=min(PER_CLASS, int(available[label])), random_state=SEED)
        for label in [0, 1]
    ]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    sampled.to_csv(OUTPUT, index=False)
    saved = pd.read_csv(OUTPUT)
    assert len(saved) == sum(min(PER_CLASS, int(n)) for n in available)
    assert saved.Text.notna().all() and not saved.Text.duplicated().any()
    assert saved.Score.isin([1, 2, 4, 5]).all()
    assert saved.sentiment.eq(saved.Score.ge(4).astype(int)).all()
    assert saved.sentiment.value_counts().to_dict() == sampled.sentiment.value_counts().to_dict()
    report["sampled_per_class"] = saved.sentiment.value_counts().sort_index().to_dict()
    report["output_rows"] = len(saved)
    REPORTS.mkdir(exist_ok=True)
    colors = ["#C85A54", "#328C82"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, counts, title in zip(axes, [available, saved.sentiment.value_counts().reindex([0, 1], fill_value=0)], ["Eligible unique reviews", "Sampled reviews (seed=42)"]):
        bars = ax.bar(["Negative (0)", "Positive (1)"], counts, color=colors)
        ax.bar_label(bars, fmt="%d", padding=3)
        ax.set(title=title, ylabel="Reviews", ylim=(0, max(1, counts.max()) * 1.15))
    fig.savefig(REPORTS / "sentiment_distribution.png", dpi=160)
    plt.close(fig)
    lengths = pd.DataFrame({"Characters": saved.Text.str.len(), "Words": saved.Text.str.split().str.len()})
    report["sample_review_lengths"] = lengths.describe().to_dict()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, unit in zip(axes, lengths.columns):
        for label, color in enumerate(colors):
            ax.hist(lengths.loc[saved.sentiment.eq(label), unit], bins=50, alpha=0.55, color=color, label=["Negative", "Positive"][label])
        ax.set(title=f"Sample review length: {unit.lower()}", xlabel=unit, ylabel="Reviews (log scale)", yscale="log")
        ax.legend()
    fig.savefig(REPORTS / "review_lengths.png", dpi=160)
    plt.close(fig)
    assert digest(SOURCE) == source_hash, "Source changed during processing"
    report["source_unchanged"] = True
    (REPORTS / "eda_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["rows", "nulls", "duplicate_rows", "duplicate_review_texts", "rating_distribution", "available_per_class", "sampled_per_class", "source_unchanged"]}, indent=2))


if __name__ == "__main__":
    main()
