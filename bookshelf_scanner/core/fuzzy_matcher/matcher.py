import duckdb
import json

from dataclasses import dataclass
from functools   import cache
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
    texts   : list[str]   # Combined texts extracted from OCR
    matches : list[dict]  # List of {title, author, score} dictionaries

class FuzzyMatcher:
    """
    Matches OCR-extracted text against a reference database of book titles.
    Uses fuzzy matching to account for OCR imperfections.
    """
    PROJECT_ROOT      = Utils.find_root('pyproject.toml')
    REFERENCE_DB_PATH = PROJECT_ROOT / 'bookshelf_scanner' / 'data' / 'books.duckdb'
    OCR_RESULTS_PATH  = PROJECT_ROOT / 'bookshelf_scanner' / 'data' / 'results' / 'extractor.json'
    OUTPUT_FILE       = PROJECT_ROOT / 'bookshelf_scanner' / 'data' / 'results' / 'matcher.json'

    def __init__(
        self,
        reference_db_path  : Path | None = None,
        ocr_results_path   : Path | None = None,
        output_file        : Path | None = None,
        max_matches        : int         = 3,
        min_ocr_confidence : float       = 0.1,
        min_match_score    : float       = 0.8
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
        self.book_records       = []
        self.candidate_strings  = []

    def load_book_records(self):
        """
        Loads book records from the database into memory and prepares candidate strings for matching.
        """
        conn    = duckdb.connect(str(self.reference_db_path))
        records = conn.execute("SELECT title, author FROM books").fetchall()
        conn.close()

        self.book_records = [BookRecord(title = title, author = author) for title, author in records]
        logger.info(f"Loaded {len(self.book_records)} book records from database")

        # For each book record, create combinations of title and author
        self.candidate_strings = [
            self.preprocess_text(f"{record.title} {record.author}")
            for record in self.book_records
        ]

    @staticmethod
    @cache
    def preprocess_text(text : str) -> str:
        """
        Preprocesses text for fuzzy matching by normalizing whitespace,
        removing punctuation, and converting to lowercase.
        """
        return fuzz_utils.default_process(text)

    def combine_texts(self, texts : list[tuple[str, float]]) -> list[str]:
        """
        Combines all texts into a single string, filtering by confidence threshold.

        Args:
            texts: List of (text, confidence) tuples from OCR

        Returns:
            List of text strings that passed the confidence threshold
        """
        filtered_texts = [
            text for text, conf in texts
            if conf >= self.min_ocr_confidence
        ]
        
        if filtered_texts:
            logger.info(f"Combined {len(filtered_texts)} text segments: {filtered_texts}")
        return filtered_texts

    def match_text(self, texts : list[str]) -> list[dict[str, Any]]:
        """
        Matches a combination of text strings against book records.
        Uses process.extract to find the best matches.

        Args:
            texts: List of text strings to match

        Returns:
            List of match dictionaries with scores above min_match_score
        """
        search_string = ' '.join(texts)
        search_string = self.preprocess_text(search_string)
        logger.info(f"Matching preprocessed string: '{search_string}'")

        # Use fuzz.token_set_ratio to reduce sensitivity to word order
        matches = process.extract(
            query   = search_string,
            choices = self.candidate_strings,
            scorer  = fuzz.token_set_ratio,
            limit   = self.max_matches * self.max_matches # Get more matches to filter later
        )

        # Build results using list comprehension
        results = [
            {
                "title"  : self.book_records[idx].title,
                "author" : self.book_records[idx].author,
                "score"  : score / 100.0
            }
            for _, score, idx in matches
            if (score / 100.0) >= self.min_match_score
        ][:self.max_matches]

        # Log each match in detail
        if results:
            for match in results:
                logger.info(
                    f"  Match found: '{match['title']}' by {match['author']} "
                    f"(Score: {match['score']:.3f})"
                )
        else:
            logger.info("  No matches found above score threshold")

        return results

    def match_books(self):
        """
        Processes all OCR results and finds matching book titles.
        Combines detected strings and saves results to JSON file.
        """
        if not self.ocr_results_path.exists():
            logger.error(f"OCR results file not found at {self.ocr_results_path}")
            return

        with open(self.ocr_results_path, 'r') as f:
            ocr_results = json.load(f)

        self.load_book_records()
        match_results = {}
        total_images  = len(ocr_results)

        logger.info(f"Processing {total_images} images")

        for image_num, (image_name, ocr_texts) in enumerate(ocr_results.items(), 1):
            logger.info(f"\nProcessing image {image_num}/{total_images}: {image_name}")
            
            texts = self.combine_texts(ocr_texts)
            if not texts:
                logger.info("No valid texts found")
                continue

            matches = self.match_text(texts)
            if matches:
                match_results[image_name] = {
                    'texts'   : texts,
                    'matches' : matches
                }
            
        logger.info(f"\nProcessing complete. Found matches for {len(match_results)} images")

        with self.output_file.open('w', encoding = 'utf-8') as f:
            json.dump(match_results, f, ensure_ascii = False, indent = 4)

        logger.info(f"Match results saved to {self.output_file}")