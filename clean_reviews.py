"""Lightweight text cleaning. Run directly to write a NEW cleaned CSV."""

import argparse
import csv
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TAGS = re.compile(r"<!--.*?-->|</?[a-z][^>]*>", re.I | re.S)
URLS = re.compile(r"\b(?:https?://|www\.)[^\s<>]+", re.I)
PUNCTUATION = re.compile(r"[^\w\s']|_", re.UNICODE)


def clean_text(text: object) -> str:
    """Return normalized text; missing/non-string inputs become empty strings.

    Keep all words (no stopword removal). Expand common negative contractions
    before removing punctuation so their negation remains explicit.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = html.unescape(text).lower().replace("’", "'")
    text = TAGS.sub(" ", text)
    text = URLS.sub(" ", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"\bshan't\b", "shall not", text)
    text = re.sub(r"\bcannot\b", "can not", text)
    text = re.sub(r"n['’]t\b", " not", text)
    text = PUNCTUATION.sub(" ", text)
    # Preserve internal apostrophes in ordinary contractions, remove quotes.
    text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)
    return " ".join(text.split())


def clean_csv(source: Path, output: Path) -> int:
    """Append clean_text, retaining every original field, label and row order.

    Exclusive creation refuses to overwrite any existing output dataset.
    """
    with source.open(encoding="utf-8", newline="") as incoming:
        reader = csv.DictReader(incoming)
        columns = reader.fieldnames
        if not columns or not {"Text", "sentiment"}.issubset(columns):
            raise ValueError("Input must contain Text and sentiment columns")
        if "clean_text" in columns:
            raise ValueError("Input already contains a clean_text column")
        with output.open("x", encoding="utf-8", newline="") as outgoing:
            writer = csv.DictWriter(outgoing, fieldnames=[*columns, "clean_text"])
            writer.writeheader()
            count = 0
            for row in reader:
                row["clean_text"] = clean_text(row["Text"])
                writer.writerow(row)
                count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed_reviews.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/cleaned_reviews.csv")
    args = parser.parse_args()
    print(f"Saved {clean_csv(args.input, args.output):,} reviews to {args.output}")
