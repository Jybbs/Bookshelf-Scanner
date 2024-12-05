import duckdb
import itertools
import json

from dataclasses import dataclass
from pathlib     import Path
from rapidfuzz   import fuzz, process, utils as fuzz_utils
from typing      import Any, NamedTuple

from bookshelf_scanner import ModuleLogger, Utils
logger = ModuleLogger('matcher')()

class BookRecord(NamedTuple):
    """
    Represents a book record from the database.
    """
    title  : str
    author : str

@dataclass
class MatchResult:
    """
    Stores the results of a text matching operation.
    """
    texts      : list[str]   # Combined texts extracted from OCR
    confidence : float       # Average OCR confidence score
    matches    : list[dict]  # List of {title, author, score} dictionaries

class FuzzyMatcher:
    """
    Matches OCR-extracted text against a reference database of book titles.
    Uses fuzzy matching to account for OCR imperfections and considers
    permutations of all text segments as potential parts of the same title.
    """
    PROJECT_ROOT      = Utils.find_root('pyproject.toml')
    REFERENCE_DB_PATH = PROJECT_ROOT / 'bookshelf_scanner' / 'data' / 'books.duckdb'
    OCR_RESULTS_PATH  = PROJECT_ROOT / 'bookshelf_scanner' / 'core' / 'text_extractor' / 'ocr_results.json'
    OUTPUT_FILE       = PROJECT_ROOT / 'bookshelf_scanner' / 'core' / 'fuzzy_matcher' / 'match_results.json'

    def __init__(
        self,
        reference_db_path  : Path | None = None,
        ocr_results_path   : Path | None = None,
        output_file        : Path | None = None,
        max_matches        : int         = 5,
        min_ocr_confidence : float       = 0.5,
        min_match_score    : float       = 0.7
    ):
        """
        Initializes the FuzzyMatcher instance.

        Args:
            reference_db_path  : Optional custom path to the books database
            ocr_results_path   : Optional custom path to OCR results file
            output_file        : Optional custom path for match results output
            max_matches        : Maximum number of matches to return per text combination
            min_ocr_confidence : Minimum confidence threshold for OCR results (0.0 to 1.0)
            min_match_score    : Minimum score threshold for fuzzy matches (0.0 to 1.0)
        """
        self.reference_db_path  = reference_db_path or self.REFERENCE_DB_PATH
        self.ocr_results_path   = ocr_results_path  or self.OCR_RESULTS_PATH
        self.output_file        = output_file       or self.OUTPUT_FILE
        self.max_matches        = max_matches
        self.min_ocr_confidence = min_ocr_confidence
        self.min_match_score    = min_match_score
        self.candidate_strings  = []
        self.book_records       = []

    def load_book_records(self) -> None:
        """
        Loads book records from the database into memory and prepares candidate strings for matching.
        """
        conn    = duckdb.connect(str(self.reference_db_path))
        records = conn.execute("SELECT title, author FROM books").fetchall()
        conn.close()

        self.book_records = [BookRecord(title = title, author = author) for title, author in records]

        # Prepare candidate strings for matching
        self.candidate_strings = [
            self.preprocess_text(s)
            for record in self.book_records
            for s in [
                f"{record.title} {record.author}",
                f"{record.author} {record.title}",
                record.title,
                record.author
            ]
        ]

    def preprocess_text(self, text : str) -> str:
        """
        Preprocesses text for fuzzy matching by normalizing whitespace,
        removing punctuation, and converting to lowercase.
        """
        return fuzz_utils.default_process(text)

    def combine_texts(self, texts : list[tuple[str, float]]) -> list[tuple[list[str], float]]:
        """
        Generates all permutations of the text strings,
        using all texts, and calculates average confidence.

        Args:
            texts : List of (text, confidence) tuples from OCR

        Returns:
            List of (combined_texts, avg_confidence) tuples
        """
        filtered_texts = [
            (text, conf) for text, conf in texts
            if conf >= self.min_ocr_confidence
        ]

        combinations = []
        if filtered_texts:
            all_texts        = filtered_texts
            perms            = list(itertools.permutations(all_texts))
            avg_confidence   = sum(conf for _, conf in filtered_texts) / len(filtered_texts)
            combinations     = [
                ([text for text, _ in perm], avg_confidence) for perm in perms
            ]
        return combinations

    def match_text(self, texts : list[str]) -> list[dict[str, Any]]:
        """
        Matches a combination of text strings against book records.
        Uses process.extract to find the best matches.

        Args:
            texts : List of text strings to match

        Returns:
            List of match dictionaries with scores above min_match_score
        """
        search_string = ' '.join(texts)
        search_string = self.preprocess_text(search_string)

        matches = process.extract(
            query   = search_string,
            choices = self.candidate_strings,
            scorer  = fuzz.token_sort_ratio,
            limit   = self.max_matches * 2  # Get more matches to filter later
        )

        results = []
        for match_string, score, idx in matches:
            match_score = score / 100.0

            if match_score >= self.min_match_score:
                book_record = self.book_records[idx % len(self.book_records)]
                results.append({
                    "title"  : book_record.title,
                    "author" : book_record.author,
                    "score"  : match_score
                })
        return results[:self.max_matches]

    def match_books(self) -> None:
        """
        Processes all OCR results and finds matching book titles.
        Combines multiple detected strings and saves results to JSON file.
        """
        if not self.ocr_results_path.exists():
            logger.error(f"OCR results file not found at {self.ocr_results_path}")
            return

        with open(self.ocr_results_path, 'r') as f:
            ocr_results = json.load(f)

        self.load_book_records()

        match_results = {}

        for image_name, ocr_texts in ocr_results.items():
            logger.info(f"Processing image: {image_name}")
            image_match_dict  = {}  # {(title, author): {'score': score, 'texts': texts, 'confidence': confidence}}
            text_combinations = self.combine_texts(ocr_texts)
            num_permutations  = len(text_combinations)
            logger.info(f"Number of permutations to explore: {num_permutations}")

            for texts, confidence in text_combinations:
                logger.info(f"Matching with combined texts: {texts} (Confidence: {confidence:.2f})")
                matches = self.match_text(texts)
                for match in matches:
                    key = (match['title'], match['author'])
                    if key not in image_match_dict or match['score'] > image_match_dict[key]['score']:
                        image_match_dict[key] = {
                            'score'      : match['score'],
                            'texts'      : texts,
                            'confidence' : confidence
                        }
                        logger.info(f"Found match: {match['title']} by {match['author']} (Score: {match['score']:.2f})")

            if image_match_dict:
                # Convert the match dict to a list
                image_matches = [
                    {
                        'title'      : title,
                        'author'     : author,
                        'score'      : info['score'],
                        'texts'      : info['texts'],
                        'confidence' : info['confidence']
                    }
                    for (title, author), info in image_match_dict.items()
                ]
                # Sort matches by score
                image_matches.sort(key = lambda x: x['score'], reverse = True)
                match_results[image_name] = image_matches
                logger.info(f"Found {len(image_matches)} matches for image {image_name}")
            else:
                logger.info(f"No matches found for image {image_name}")

        with self.output_file.open('w', encoding = 'utf-8') as f:
            json.dump(match_results, f, ensure_ascii = False, indent = 4)

        logger.info(f"Match results saved to {self.output_file}")
