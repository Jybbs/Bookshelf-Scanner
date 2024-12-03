import duckdb
import json

from dataclasses   import dataclass
from pathlib       import Path
from rapidfuzz     import process, fuzz

from bookshelf_scanner import ModuleLogger, Utils
logger = ModuleLogger('matcher')()

@dataclass
class MatchResult:
    """
    Stores the results of a text matching operation.
    """
    extracted_text : str                    # Text extracted from OCR
    confidence     : float                  # OCR confidence score
    matches        : list[tuple[str, int]]  # List of (matched_title, match_score) tuples

class FuzzyMatcher:
    """
    Matches OCR-extracted text against a master database of book titles.
    Uses fuzzy matching to account for OCR imperfections.
    """
    PROJECT_ROOT     = Utils.find_root('pyproject.toml')
    MASTER_DB_PATH   = PROJECT_ROOT / 'bookshelf_scanner' / 'data'  / 'books.duckdb'
    OCR_RESULTS_PATH = PROJECT_ROOT / 'bookshelf_scanner' / 'core' / 'text_extractor' / 'ocr_results.json'
    OUTPUT_FILE      = PROJECT_ROOT / 'bookshelf_scanner' / 'core' / 'matcher' / 'match_results.json'

    def __init__(
        self,
        master_db_path   : Path | None = None,
        ocr_results_path : Path | None = None,
        output_file      : Path | None = None,
        match_threshold  : int         = 80,
        max_matches      : int         = 3
    ):
        """
        Initializes the FuzzyMatcher instance.

        Args:
            master_db_path   : Optional custom path to the books database
            ocr_results_path : Optional custom path to OCR results file
            output_file      : Optional custom path for match results output
            match_threshold  : Minimum score for considering a match valid
            max_matches      : Maximum number of matches to return per text
        """
        self.master_db_path   = master_db_path   or self.MASTER_DB_PATH
        self.ocr_results_path = ocr_results_path or self.OCR_RESULTS_PATH
        self.output_file      = output_file      or self.OUTPUT_FILE
        self.match_threshold  = match_threshold
        self.max_matches      = max_matches
        
        self.validate_paths()

    def validate_paths(self) -> None:
        """
        Validates that required input files exist.
        
        Raises:
            FileNotFoundError: If required files are not found
        """
        if not self.master_db_path.exists():
            raise FileNotFoundError(f"Master database not found at {self.master_db_path}")
            
        if not self.ocr_results_path.exists():
            raise FileNotFoundError(f"OCR results file not found at {self.ocr_results_path}")
            
        self.output_file.parent.mkdir(parents = True, exist_ok = True)

    def load_book_titles(self) -> list[str]:
        """
        Retrieves book titles from the master database.
        
        Returns:
            List of book titles from database
        """
        conn = duckdb.connect(str(self.master_db_path))
        try:
            result = conn.execute("SELECT title FROM books").fetchall()
            return [title[0] for title in result]
        finally:
            conn.close()

    def load_ocr_results(self) -> dict[str, list[tuple[str, float]]]:
        """
        Loads OCR results from JSON file.
        
        Returns:
            Dictionary mapping image names to lists of (text, confidence) tuples
        """
        with self.ocr_results_path.open('r') as f:
            return json.load(f)

    def find_matches(
        self,
        extracted_text : str,
        book_titles    : list[str]
    ) -> list[tuple[str, int]]:
        """
        Performs fuzzy matching between extracted text and book titles.
        
        Args:
            extracted_text : Text extracted from OCR
            book_titles   : List of book titles to match against
            
        Returns:
            List of (title, score) tuples for best matches
        """
        matches = process.extract(
            query   = extracted_text,
            choices = book_titles,
            scorer  = fuzz.ratio,
            limit   = self.max_matches
        )
        
        return [(title, score) for title, score in matches if score >= self.match_threshold]

    def match_books(self) -> None:
        """
        Processes all OCR results and finds matching book titles.
        Saves results to JSON file.
        """
        book_titles = self.load_book_titles()
        ocr_results = self.load_ocr_results()
        all_matches = {}
        
        for image_name, texts in ocr_results.items():
            image_matches = []
            
            for extracted_text, confidence in texts:
                matches = self.find_matches(extracted_text, book_titles)
                
                if matches:
                    match_result = MatchResult(
                        extracted_text = extracted_text,
                        confidence     = confidence,
                        matches        = matches
                    )
                    image_matches.append(match_result.__dict__)
                    
                    logger.info(
                        f"Matched '{extracted_text}' (conf: {confidence:.2f}) "
                        f"to {len(matches)} titles"
                    )
            
            if image_matches:
                all_matches[image_name] = image_matches
        
        with self.output_file.open('w', encoding = 'utf-8') as f:
            json.dump(all_matches, f, ensure_ascii = False, indent = 4)
            
        logger.info(f"Match results saved to {self.output_file}")