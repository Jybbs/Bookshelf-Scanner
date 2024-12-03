# Fuzzy Matcher

The Fuzzy Matcher component performs intelligent text matching between OCR-extracted book spine text and a master database of book titles. It employs fuzzy string matching algorithms to handle OCR imperfections, text variations, and partial matches.

## Overview

This component processes the output from the Text Extractor module, matching each extracted text segment against a comprehensive database of known book titles. The fuzzy matching approach ensures that minor OCR errors or formatting differences don't prevent successful matches.

## Core Features

The matcher provides sophisticated text matching capabilities with:

- Configurable matching thresholds for fine-tuning accuracy
- Adjustable match limits per text segment
- Confidence scoring for each potential match
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
    match_threshold = 85,    # Minimum score to consider a match valid
    max_matches     = 3       # Maximum number of matches per text segment
)
matcher.match_books()
```

## Configuration

The matcher supports several configuration options:

```python
FuzzyMatcher(
    master_db_path   = Path("path/to/custom/database.duckdb"),
    ocr_results_path = Path("path/to/custom/results.json"),
    output_file      = Path("path/to/custom/output.json"),
    match_threshold  = 80,
    max_matches      = 3
)
```

## Output Format

The matcher generates a JSON file structured as follows:

```json
{
    "bookshelf_image_1.jpg": [
        {
            "extracted_text": "Foundation",
            "confidence": 0.92,
            "matches": [
                ["Foundation", 100],
                ["Foundation and Empire", 85],
                ["Second Foundation", 82]
            ]
        }
    ]
}
```

## Dependencies

This module requires:

- DuckDB for database operations
- RapidFuzz for fuzzy string matching
- Project utilities for logging and path management