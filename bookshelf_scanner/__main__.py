import argparse

from bookshelf_scanner import *
from pathlib           import Path

def main():

    parser = argparse.ArgumentParser(
        description = "Run the entire Bookshelf Scanner pipeline."
    )

    parser.add_argument(
        "--book-segmenter",
        action = "store_true",
        help   = "Run the BookSegmenter step."
    )
    parser.add_argument(
        "--config-optimizer",
        action = "store_true",
        help   = "Run the ConfigOptimizer step to find optimal OCR parameters."
    )
    parser.add_argument(
        "--fuzzy-matcher",
        action = "store_true",
        help   = "Run the FuzzyMatcher step to match OCR text to known book titles."
    )
    parser.add_argument(
        "--match-approver",
        action = "store_true",
        help   = "Run the MatchApprover UI to interactively confirm matched titles."
    )

    parser.add_argument(
        "--images_dir",
        type    = str,
        default = "images/bookcases",
        help    = "Directory containing the bookshelf images."
    )

    args       = parser.parse_args()
    images_dir = Utils.find_root('pyproject.toml') / 'bookshelf_scanner' / Path(args.images_dir)

    # If no steps were explicitly requested, run them all.
    if not (args.book_segmenter or args.config_optimizer or args.fuzzy_matcher or args.match_approver):
        args.book_segmenter   = True
        args.config_optimizer = True
        args.fuzzy_matcher    = True
        args.match_approver   = True

    # Run BookSegmenter
    if args.book_segmenter:
        segmenter = BookSegmenter(output_images = True, output_json = True)
        for image_path in images_dir.glob("*.jpg"):
            segmenter.segment_books(image_path = image_path)

    # Run ParameterOptimizer
    if args.config_optimizer:
        extractor    = TextExtractor()
        optimizer    = ConfigOptimizer(extractor = extractor, output_images = True)
        spine_images = extractor.find_image_files(subdirectory = 'books')
        optimizer.optimize(spine_images)

    # Run TextMatcher
    if args.fuzzy_matcher:
        matcher = FuzzyMatcher()
        matcher.match_books()

    # Run MatchApprover
    if args.match_approver:
        approver = MatchApprover(threshold = 0.75)
        approver.run_interactive_mode()

if __name__ == "__main__":
    main()
