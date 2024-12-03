import json
from pathlib import Path
import duckdb
from rapidfuzz import process, fuzz

def main():
    # Define paths for OCR results and master book list
    ocr_results_path = Path("bookshelf_scanner/core/text_extractor/ocr_results.json") #found text from images
    master_db_path = Path("bookshelf_scanner/data/books.duckdb") #master book list/database

    # Load OCR results from JSON - where found text is stored
    if not ocr_results_path.exists():
        print(f"OCR results file not found at {ocr_results_path}")
        return

    with open(ocr_results_path, 'r') as file:
        ocr_results = json.load(file)

    # Connect to DuckDB and retrieve book titles
    conn = duckdb.connect(str(master_db_path))
    book_titles = conn.execute("SELECT title FROM books").fetchall()
    book_titles = [title[0] for title in book_titles]

    # Loop through OCR results and perform fuzzy matching
    for image_name, ocr_texts in ocr_results.items():
        for text_info in ocr_texts:
            extracted_text, confidence = text_info
            print(f"Extracted text: '{extracted_text}' (Confidence: {confidence:.2f})")

            # Perform fuzzy matching
            matches = process.extract(extracted_text, book_titles, scorer=fuzz.ratio, limit=3)
            print(f"Top matches for extracted text '{extracted_text}':")
            for match in matches:
                print(f"  - {match[0]} (Score: {match[1]})")

# Entry point
if __name__ == "__main__":
    main()
