# Bookshelf Scanner

A computer vision system that detects and extracts text from book spines on bookshelves, built for Northeastern University's CS5330 Pattern Recognition & Computer Vision course.

## Overview

This project automates the process of capturing book information directly from bookshelf images. By processing photos of bookshelves, the system detects individual book spines and extracts visible text, enabling efficient digitization of physical book collections.

The system processes bookshelf images through three main stages:

1. **Spine Segmentation**: Uses YOLOv8 to detect and segment individual book spines from bookshelf images
2. **Text Extraction**: Processes the segmented spine images to extract visible text using EasyOCR
3. **Text Matching**: *(Planned)* Uses RapidFuzz to match extracted spine text against a book database

## Key Features

- **Automated Spine Detection**: Accurately identifies and segments individual book spines from shelf images
- **Text Extraction**: Captures visible text from book spines including:
  - Title information
  - Author names
  - Series information (*when visible*)
- **Quality Assurance**: Confidence scoring for both spine detection and text extraction
- **Bulk Processing**: Process entire bookshelves in a single operation

## Project Structure

```
├── BookSegmenter/             # Core spine detection module
│   ├── models/                # Pre-trained YOLO model directory
│   │   └── OpenShelves8.onnx  # YOLO model optimized for spine detection
│   ├── BookSegmenter_base.py  # Main segmentation interface
│   ├── YOLOv8.py              # YOLO model wrapper and inference logic
│   └── utils.py               # Segmentation utility functions
|
├── TextExtractor/             # OCR and text processing module
│   ├── TextExtractor.py       # Interactive OCR processing interface
│   ├── params.yml             # Configurable OCR and image processing parameters
│   └── TextExtractor.log      # Processing logs and debugging information
|
├── DuckDB/                    # Database integration
│   ├── books.duckdb           # Local book database for text matching
|
├── images/                    # Image asset directories
│   ├── Bookcases/             # Raw, unprocessed bookshelf photos
│   ├── Books/                 # Individual spine images after segmentation
│   └── Shelves/               # Intermediate shelf processing results
|
├── poetry.lock                # Poetry dependency lock file
└── pyproject.toml             # Project metadata and dependencies
```

## Installation

### Prerequisites

- Python 3.12
- Poetry (recommended for dependency management)
- OpenCV system dependencies:
  ```bash
  # MacOS
  brew install opencv
  ```
- ONNX Runtime dependencies (installed automatically via Poetry)

### Setup

1. Clone the repository:
```bash
git clone git@github.com:your-username/bookshelf-scanner.git
cd bookshelf-scanner
```

2. Install Poetry if you haven't already:
```bash
# MacOS
brew install poetry

# Linux/WSL
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install project dependencies:
```bash
poetry install
```

## Usage

### Spine Segmentation

The `BookSegmenter` class handles splitting bookshelf images into individual spine images. See detailed instructions in [`BookSegmenter/README.md`](./BookSegmenter/README.md).

```python
from BookSegmenter import BookSegmenter

# Initialize segmenter
segmenter = BookSegmenter()

# Process a bookshelf image
image = cv2.imread("path/to/bookshelf.jpg")
spines, bboxes, confidences = segmenter.segment(image)

# Display results
segmenter.display_segmented_books(spines, confidences)
```

### Text Extraction

The `TextExtractor` provides an interactive interface for processing spine images. See detailed instructions in [`TextExtractor/README.md`](./TextExtractor/README.md).

```python
# Run the interactive text extraction process
python TextExtractor/TextExtractor.py
```

Key features:
- Real-time parameter adjustment for image processing
- OCR text extraction with confidence scores
- Results saved to `ocr_results.json`

### Text Matching *(Planned)*

Future implementation will include:
- Fuzzy text matching using RapidFuzz for identifying books from partial spine text
- Integration with DuckDB for book information lookup
- Configurable matching thresholds for accuracy control

## Technical Details

- **Computer Vision**: Custom-trained YOLOv8 model optimized for book spine detection
- **OCR Processing**: EasyOCR with parameters tuned for text extraction with inconsistent color and shape profiles
- **Data Storage**: Efficient DuckDB integration for book information lookup

## Configuration

- Text extraction parameters can be modified in `TextExtractor/params.yml`
- Spine detection uses the `OpenShelves8.onnx` model located in `BookSegmenter/models/`

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.