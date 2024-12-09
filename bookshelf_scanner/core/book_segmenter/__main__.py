"""
Entry point for running the BookSegmenter as a module.
"""

from bookshelf_scanner import BookSegmenter, Utils

def main():

    segmenter = BookSegmenter(
        output_images = True,
        output_json   = True
    )

    image_path = Utils.find_root('pyproject.toml') / 'bookshelf_scanner' / 'images' / 'bookcases' / 'IMG_1538.jpg'
    segmenter.segment_books(image_path = image_path)

if __name__ == "__main__":
    main()
