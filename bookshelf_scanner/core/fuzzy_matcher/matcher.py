import duckdb
import json

from dataclasses import dataclass
from pathlib     import Path
from rapidfuzz   import process, fuzz

from bookshelf_scanner import ModuleLogger, Utils
logger = ModuleLogger('matcher')()

@dataclass
class MatchResult:
    """
    Stores the results of a text matching operation.
    """
    text      : str         # Text extracted from OCR
    confidence : float      # OCR confidence score
    matches    : list[dict] # List of {title, score} dictionaries

class FuzzyMatcher:
    """
    Matches OCR-extracted text against a master database of book titles.
    Uses fuzzy matching to account for OCR imperfections.
    """
    PROJECT_ROOT     = Utils.find_root('pyproject.toml')
    MASTER_DB_PATH   = PROJECT_ROOT / 'bookshelf_scanner' / 'data'  / 'books.duckdb'
    OCR_RESULTS_PATH = PROJECT_ROOT / 'bookshelf_scanner' / 'core' / 'text_extractor' / 'ocr_results.json'
    OUTPUT_FILE      = PROJECT_ROOT / 'bookshelf_scanner' / 'core' / 'fuzzy_matcher' / 'match_results.json'

    def __init__(
        self,
        master_db_path   : Path | None = None,
        ocr_results_path : Path | None = None,
        output_file      : Path | None = None,
        max_matches      : int         = 3
    ):
        """
        Initializes the FuzzyMatcher instance.

        Args:
            master_db_path   : Optional custom path to the books database
            ocr_results_path : Optional custom path to OCR results file
            output_file      : Optional custom path for match results output
            max_matches      : Maximum number of matches to return per text
        """
        self.master_db_path   = master_db_path   or self.MASTER_DB_PATH
        self.ocr_results_path = ocr_results_path or self.OCR_RESULTS_PATH
        self.output_file      = output_file      or self.OUTPUT_FILE
        self.max_matches      = max_matches

    def match_books(self):
        """
        Processes all OCR results and finds matching book titles.
        Saves results to JSON file in a format matching the original script.
        """
        if not self.ocr_results_path.exists():
            logger.error(f"OCR results file not found at {self.ocr_results_path}")
            return

        # Load OCR results
        with open(self.ocr_results_path, 'r') as f:
            ocr_results = json.load(f)

        # Get book titles from database
        conn = duckdb.connect(str(self.master_db_path))
        book_titles = conn.execute("SELECT title FROM books").fetchall()
        book_titles = [title[0] for title in book_titles]
        conn.close()

        # Process matches for each image
        match_results = {}
        
        for image_name, ocr_texts in ocr_results.items():
            image_matches = []
            
            for text_info in ocr_texts:
                extracted_text, confidence = text_info
                
                # Perform fuzzy matching
                matches = process.extract(
                    query     = extracted_text,
                    choices   = book_titles,
                    scorer    = fuzz.ratio,
                    limit     = self.max_matches
                )
                
                # Log the matches
                logger.info(f"Extracted text: '{extracted_text}' (Confidence: {confidence:.2f})")
                for match in matches:
                    logger.info(f"  - {match[0]} (Score: {match[1]})")
                
                # Store results
                result = {
                    "text"       : extracted_text,
                    "confidence" : confidence,
                    "matches"    : [
                        {
                            "title" : match[0],
                            "score" : match[1]
                        }
                        for match in matches
                    ]
                }
                image_matches.append(result)
            
            if image_matches:
                match_results[image_name] = image_matches

        # Save results to JSON
        with self.output_file.open('w', encoding = 'utf-8') as f:
            json.dump(match_results, f, ensure_ascii = False, indent = 4)
            
        logger.info(f"Match results saved to {self.output_file}")