# FuzzyMatcher

The Fuzzy Matcher component performs intelligent text matching between OCR-extracted book spine text and a reference database of book titles. It employs fuzzy string matching algorithms to handle OCR imperfections, text variations, and partial matches using RapidFuzz.

## Overview

This component processes the output from the Text Extractor module, matching each extracted text segment against a reference database of book titles and authors. The fuzzy matching approach uses token set ratio comparison to ensure that word order variations, minor OCR errors, or formatting differences don't prevent successful matches.

## Core Features

The matcher provides sophisticated text matching capabilities with:

- Configurable confidence thresholds for both OCR and match quality
- Adjustable maximum matches per text segment
- Normalized scoring system (0.0 to 1.0)
- Text preprocessing to handle punctuation and case variations
- Detailed logging of the matching process
- Structured JSON output for downstream processing

## Usage

The matcher can be run directly using Poetry:

```bash
poetry run fuzzy-matcher
```

For programmatic use within your Python code:

```python
from bookshelf_scanner import FuzzyMatcher

matcher = FuzzyMatcher(
    min_ocr_confidence = 0.1,  # Minimum confidence for OCR results
    min_match_score    = 0.8,   # Minimum score to consider a match valid
    max_matches        = 3       # Maximum number of matches per text segment
)
matcher.match_books()
```

## Configuration

The matcher supports several configuration options:

```python
FuzzyMatcher(
    reference_db_path  = Path("path/to/custom/database.duckdb"),
    ocr_results_path   = Path("path/to/custom/results.json"),
    output_file        = Path("path/to/custom/output.json"),
    max_matches        = 3,
    min_ocr_confidence = 0.1,
    min_match_score    = 0.8
)
```

## Output Format

The matcher generates a JSON file structured as follows:

```json
{
    "bookshelf_image_1.jpg": {
        "texts": [
            "Foundation",
            "Isaac Asimov"
        ],
        "matches": [
            {
                "title": "Foundation",
                "author": "Isaac Asimov",
                "score": 0.95
            },
            {
                "title": "Foundation and Empire",
                "author": "Isaac Asimov",
                "score": 0.85
            }
        ]
    }
}
```

## Dependencies

This module requires:

- DuckDB for database operations
- RapidFuzz for fuzzy string matching
- Project utilities for logging and path management