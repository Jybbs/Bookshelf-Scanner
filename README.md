# Bookshelf Scanner

A computer vision system that detects and extracts text from book spines on bookshelves, built for Northeastern University's CS5330 Pattern Recognition & Computer Vision course.

## Overview

This project automates the process of capturing book information directly from bookshelf images. By processing photos of bookshelves, the system detects individual book spines and extracts visible text, enabling efficient digitization of physical book collections.

The system processes bookshelf images through four main stages:

1. **Spine Segmentation**: Uses YOLOv8 to detect and segment individual book spines from bookshelf images
2. **Text Extraction**: Processes the segmented spine images to extract visible text using EasyOCR
3. **Parameter Optimization**: Fine-tunes processing parameters to maximize text extraction accuracy
4. **Text Matching**: Uses RapidFuzz to match extracted spine text against a book database, handling OCR imperfections and variations

Each component maintains its own log file through the `ModuleLogger` system, ensuring consistent debugging and monitoring across the project.

## Project Structure

```
├── bookshelf_scanner/
│   ├── __init__.py              # Package initialization and version info
│   ├── config/
│   │   └── params.yml           # Processing parameters and settings
│   ├── core/ 
│   │   ├── book_segmenter/      # Spine detection and segmentation
│   │   ├── fuzzy_matcher/       # Text matching against book database
│   │   ├── module_logger/       # Standardized logging configuration
│   │   ├── parameter_optimizer/ # Processing parameter optimization
│   │   ├── text_extractor/      # OCR and text processing
│   │   └── utils/               # Core project utilities and path handling
│   ├── data/
│   │   ├── books.duckdb         # Local book DuckDB database
│   │   └── utils/               # Data handling utilities
│   ├── dev/
│   ├── logs/                    # Component-specific log files
│   └── images/ 
│       ├── Bookcases/           # Raw bookshelf photos
│       ├── Books/               # Segmented spine images
│       └── Shelves/             # Processing results
├── poetry.lock                  # Poetry dependency lock file
└── pyproject.toml               # Project metadata and dependencies
```

## Installation

### Prerequisites

- Python 3.12
- Poetry (*recommended for dependency management*)
- OpenCV system dependencies:
  ```bash
  # MacOS
  brew install opencv
  ```
- ONNX Runtime dependencies (*installed automatically via Poetry*)

### Setup

1. Clone the repository:
```bash
git clone git@github.com:your-username/bookshelf-scanner.git
cd bookshelf_scanner
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

### Importing Components

```python
from bookshelf_scanner import BookSegmenter, TextExtractor, ParameterOptimizer, FuzzyMatcher

# Initialize components
segmenter = BookSegmenter()
extractor = TextExtractor(gpu_enabled = True)
optimizer = ParameterOptimizer(extractor)
matcher   = FuzzyMatcher(match_threshold = 85)
```

### Spine Segmentation

```python
# Process a bookshelf image
image = cv2.imread("path/to/bookshelf.jpg")
spines, bboxes, confidences = segmenter.segment(image)

# Display results
segmenter.display_segmented_books(spines, confidences)
```

### Pre-Processing & Text Extraction

The `TextExtractor` provides an interactive environment for processing spine images and extracting text. It implements multiple image processing techniques that can be adjusted in real-time to handle different image qualities and characteristics.

```python
# Process images with specific parameters
results = extractor.interactive_experiment(
    image_files     = spine_images,
    params_override = {
        'shadow_removal' : {
            'enabled'    : True,
            'parameters' : {
                'shadow_kernel_size' : 23,
                'shadow_median_blur' : 15
            }
        },

        'color_clahe' : {
            'enabled'    : True,
            'parameters' : {
                'clahe_clip_limit' : 2.0
            }
        }
    }
)
```

### Parameter Optimization

```python
# Run optimization with resume capability
optimal_params = optimizer.optimize(
    image_files    = spine_images,
    batch_size     = 100,
    save_frequency = 10,
    resume         = True
)
```

### Text Matching

The `FuzzyMatcher` component matches the optimized OCR results against a database of known book titles:

```python
# Match extracted text to book database
matcher = FuzzyMatcher(
    match_threshold = 85, # Minimum score for valid matches
    max_matches     = 3   # Maximum matches per text segment
)
matcher.match_books()

# Run via Poetry command
poetry run fuzzy-matcher
```

For detailed component documentation, see:
- [BookSegmenter Documentation](./bookshelf_scanner/core/book_segmenter/README.md)
- [TextExtractor Documentation](./bookshelf_scanner/core/text_extractor/README.md)
- [ParameterOptimizer Documentation](./bookshelf_scanner/core/parameter_optimizer/README.md)
- [FuzzyMatcher Documentation](./bookshelf_scanner/core/fuzzy_matcher/README.md)
- [ModuleLogger Documentation](./bookshelf_scanner/core/module_logger/README.md)

## Configuration

- Processing parameters can be modified in `bookshelf_scanner/config/params.yml`
- Component-specific settings are documented in their respective README files

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.