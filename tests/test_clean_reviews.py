import csv
from pathlib import Path
import tempfile
import unittest

from clean_reviews import clean_csv, clean_text


class CleanTextTests(unittest.TestCase):
    def test_html_urls_and_spacing(self):
        self.assertEqual(clean_text('<b>GREAT</b><br> food!!! https://example.com/a?q=1  www.example.org'), 'great food')

    def test_negation(self):
        self.assertEqual(clean_text("Not good; no flavor. NEVER again! I can't, won't, shouldn't."),
                         'not good no flavor never again i can not will not should not')

    def test_missing_and_empty(self):
        for value in [None, float('nan'), '', ' \t\n', 123]:
            with self.subTest(value=value):
                self.assertEqual(clean_text(value), '')

    def test_entities_unicode_and_apostrophes(self):
        self.assertEqual(clean_text("&lt;b&gt;CAFÉ&lt;/b&gt; isn’t 'bad' &amp; it's fresh."),
                         "café is not bad it's fresh")

    def test_idempotence(self):
        text = "<p>No!!! It's NOT bad.</p> https://example.com"
        self.assertEqual(clean_text(clean_text(text)), clean_text(text))

    def test_csv_preservation_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.csv'
            output = Path(directory) / 'cleaned.csv'
            source.write_text('Text,sentiment,Id\n"NOT good!",0,001\n,1,002\n', encoding='utf-8')
            original = source.read_bytes()
            self.assertEqual(clean_csv(source, output), 2)
            with output.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows, [
                {'Text': 'NOT good!', 'sentiment': '0', 'Id': '001', 'clean_text': 'not good'},
                {'Text': '', 'sentiment': '1', 'Id': '002', 'clean_text': ''},
            ])
            saved = output.read_bytes()
            with self.assertRaises(FileExistsError):
                clean_csv(source, output)
            self.assertEqual(output.read_bytes(), saved)
            with self.assertRaises(FileExistsError):
                clean_csv(source, source)
            self.assertEqual(source.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
