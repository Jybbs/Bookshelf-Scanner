import json
import csv
import os

from bookshelf_scanner import Utils

def convert_matcher_to_csv(json_file_path, output_csv_path):
    try:
        # Read the matcher.json file
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

        # Open the CSV file for writing
        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)

            # Write the headers
            writer.writerow(["filename", "title", "author", "confidence score (%)"])

            # Process each file's data
            for filename, content in data.items():
                matches = content.get("matches", [])
                for match in matches:
                    title = match.get("title", "N/A")
                    author = match.get("author", "N/A")
                    score = match.get("score", 0) * 100  # Convert to percentage

                    # Write the row
                    writer.writerow([filename, title, author, f"{score:.1f}%"])

        print(f"CSV file successfully created at: {output_csv_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# File paths (adjust as needed)
PROJECT_ROOT      = Utils.find_root('pyproject.toml')
MATCHER_JSON_PATH = PROJECT_ROOT / 'bookshelf_scanner' / 'data' / 'results' / 'matcher.json'
CSV_RESULTS       = PROJECT_ROOT / 'bookshelf_scanner' / 'data' / 'results' / 'approvals.csv'

# Convert the JSON to CSV
convert_matcher_to_csv(MATCHER_JSON_PATH, CSV_RESULTS)
