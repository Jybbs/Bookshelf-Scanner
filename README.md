# Bookshelf Scanner

**Bookshelf Scanner** is a computer vision project focused on extracting and organizing text from book spines in bookshelf images. It integrates object detection, image preprocessing, optical character recognition (**OCR**), parameter optimization, and fuzzy text matching, aiming to reduce manual effort and trial-and-error commonly required for such tasks.

This pipeline addresses various challenges: the complexity of detecting books in cluttered scenes, dealing with inconsistent lighting and low-contrast text, finding effective preprocessing parameters for OCR, and mapping imperfectly recognized text to known book titles.

---

## Overview of the Pipeline

The system operates in stages:

1. **Spine Segmentation**:  
   A YOLOv8 model locates individual book spines in a shelf image. Each detected spine is cropped out, producing a set of spine images ready for further processing. This step isolates the relevant portion of the image, ensuring that subsequent steps work on just the book region rather than the entire shelf.

2. **Preprocessing & Text Extraction**:  
   The `TextExtractor` module applies a series of adjustable image enhancements to these spine images before running OCR:

   - **Shadow Removal**: Attempts to correct uneven illumination, often caused by bookshelf shadows or uneven lighting. By normalizing brightness, text stands out more clearly.

   - **Color CLAHE (*Contrast Limited Adaptive Histogram Equalization*)**: Improves local contrast without excessively boosting noise. This is helpful for faint text and subtle color variations.
   - **Brightness/Contrast Adjustments**: Allows fine-tuning of overall image intensity and dynamic range. Underexposed or low-contrast text can become more legible.
   - **Rotation**: Correcting the image orientation ensures text lines are horizontal or vertical, aiding OCR tools in reading characters accurately.

   This module can run in:
   
   - **Interactive Mode**: A graphical interface lets you enable/disable steps (*like shadow removal or CLAHE*) and adjust parameters (*like brightness increments*) on-the-fly. As you tweak values, you immediately see changes reflected in OCR results. This helps understand parameter sensitivity and guides initial parameter guesses.

   - **Headless Mode**: Processes images in batches without a UI, using given parameters or previously found good configurations. Suitable for large-scale or automated runs once you’ve established settings that generally work well.

   After preprocessing, EasyOCR extracts text lines and confidence scores. If the text is initially unreadable at a certain orientation, the system can try alternative rotations and pick the best result.

3. **Parameter Configuration Optimization**:  
   Choosing effective preprocessing parameters can be challenging. The `ConfigOptimizer` helps automate this search:

   - **Parameter Space**: Each preprocessing step has parameters (*e.g., kernel sizes for shadow removal, clip limits for CLAHE, brightness offsets, rotation angles*). The combination forms a multi-dimensional parameter space.
  
   - **Model-Based Search**: Instead of brute-forcing every parameter combination or manually guessing, the optimizer learns a surrogate model that approximates how parameters affect OCR performance.  
   - **Uncertainty Estimation**: It estimates uncertainty in predictions by performing multiple stochastic forward passes (*e.g., with dropout*) to obtain a distribution of predicted OCR quality.  
   - **Acquisition Function (*UCB*)**: With mean and variance of predictions, it applies a selection criterion like Upper Confidence Bound (**UCB**) to pick the next parameter set to evaluate. This balances testing known good regions (*exploitation*) and exploring new areas that might yield even better results (*exploration*).
   - **Outcome**: Over multiple iterations, the optimizer converges toward a set of parameters that improve OCR results, reducing the need for manual tuning. It does not guarantee a global optimum, but it aims to find a reasonably good configuration more efficiently than naive methods.

4. **Text Matching**:  
   Once text is extracted, the `FuzzyMatcher` module compares it to a database of known books. OCR may produce incomplete or slightly incorrect text. Fuzzy matching tolerates such imperfections. It can handle partial matches, character substitutions, and word order variations. This step links OCR output to actual titles, making the extracted data more meaningful and searchable.

**Logging with ModuleLogger**:  
Throughout, a `ModuleLogger` system provides consistent, module-specific logging. Each component records operations, parameter changes, and results, aiding in debugging, monitoring performance, and maintaining reproducibility.

---

## Technical Highlights

- **Spine Detection (*YOLOv8*)**:  
  Utilizes a trained model to identify bounding boxes around spines. The model’s accuracy and generalization depend on the training data. This step transforms a complex scene into a set of targeted sub-images.

- **Preprocessing Steps Detail**:

  - **Shadow Removal**: Involves morphological operations and median filters. For example, a dilate-then-subtract approach can highlight text while minimizing shadow gradients.

  - **CLAHE**: Applied to the luminance channel in a LAB color space, CLAHE avoids global overexposure. Parameters like `clip_limit` control how aggressively contrast is stretched.
  - **Brightness/Contrast**: Simple arithmetic and scaling on pixel intensities. Incrementing brightness shifts intensity upward, while contrast scaling widens or narrows the intensity distribution.
  - **Rotation**: Usually done in multiples of 90°, making it straightforward and lossless. Automatically trying multiple rotations increases the chance of readable text lines.

- **ConfigOptimizer Mechanics**:

  - **Surrogate Model**: Gathers data (*parameter sets and resulting OCR scores*) as it goes. With each evaluation, it updates an internal model (*often a neural network*) that predicts OCR performance from parameters.

  - **Sampling with Uncertainty**: By enabling dropout or similar techniques at inference time, the model produces different predictions per pass, approximating a distribution of possible outcomes. This helps gauge which regions of parameter space are well-understood (*low uncertainty*) and which might still hold potential (*high uncertainty*).
  - **UCB Acquisition**: The optimizer picks parameters to try next based on both mean predicted performance and uncertainty. It may try parameter settings in unexplored regions if uncertainty is high, aiming to discover better configurations.

- **Fuzzy Matching**:

  - Uses tokenization and character-based similarity metrics to handle OCR's partial errors.

  - Maps extracted strings (*which may be incomplete or slightly misspelled*) to known book titles with a similarity score. Higher scores indicate closer matches.

---

## Project Structure

```
├── bookshelf_scanner/
│   ├── config/
│   │   └── params.yml        # Default parameter settings (*e.g., min/max for brightness*)
│   ├── core/
│   │   ├── book_segmenter/   # Spine detection code
│   │   ├── fuzzy_matcher/    # Fuzzy text matching
│   │   ├── module_logger/    # Logging utilities
│   │   ├── config_optimizer/ # ConfigOptimizer and related code
│   │   ├── text_extractor/   # OCR and preprocessing logic
│   │   └── utils/            # Common utilities
│   ├── data/
│   │   ├── results/          # OCR outputs, optimization results, etc.
│   │   ├── books.duckdb      # Local DB of known books
│   │   └── utils/            # Database utility scripts
│   ├── logs/                 # Logs per module
│   └── images/               # Bookcase photos, segmented spines, etc.
├── pyproject.toml
└── poetry.lock
```

---

## Installation

**Requirements**:

- Python 3.12

- Poetry for dependency management
- OpenCV system dependencies (install via system package manager)
- ONNX Runtime (installed by Poetry)

**Setup**:
```bash
git clone git@github.com:your-username/bookshelf-scanner.git
cd bookshelf_scanner
poetry install
```

---

## Usage

Below are detailed usage scenarios for each major component, illustrating how to integrate and leverage their functionalities.

### Spine Segmentation

```python
import cv2
from bookshelf_scanner import BookSegmenter

segmenter = BookSegmenter()

# Load a full bookshelf image
image = cv2.imread("path/to/bookshelf.jpg")

# Detect and segment out individual book spines
spines, bboxes, confidences = segmenter.segment(image)

# Optionally display the segmented books
segmenter.display_segmented_books(spines, confidences)
```

**What This Does**: 

- Detects each spine and crops it out as a separate image in `spines`. 
 
- Provides bounding boxes `bboxes` and detection `confidences`.  
- After this step, you have a list of spine images ready for preprocessing and OCR.

### Text Extraction (*Interactive or Headless*)

```python
from bookshelf_scanner import TextExtractor

extractor   = TextExtractor(gpu_enabled = True)
image_files = extractor.find_image_files(subdirectory = 'Books')
```

**Interactive Mode**:
```python
# Launch a UI window where you can toggle steps (1, 2, 3...) and adjust parameters (e.g., B/b for brightness)
extractor.interactive_mode(image_files = image_files)
```

In the interactive window:  
- Press number keys to enable/disable specific preprocessing steps (*e.g., shadow removal, CLAHE*).

- Press uppercase/lowercase parameter keys to increase/decrease parameters like brightness or rotation angle.
- Immediately see how OCR results change (*displayed in logs or annotated on the preview*).

**Headless Mode**:
```python
# For automated batch processing, no UI. Just run and get results.
extractor.run_headless_mode(image_files = image_files)
```

You can override parameters programmatically:
```python
config_override = {
    'shadow_removal': {
        'enabled'    : True,
        'parameters' : {
            'shadow_kernel_size' : 23,
            'shadow_median_blur' : 15
        }
    }
}

extractor.run_headless_mode(
  image_files     = image_files, 
  config_override = config_override
)
```

**What This Does**: 

- Preprocesses each spine image to improve OCR accuracy.

- Extracts text using EasyOCR.
- (*Interactive mode*) Lets you visually refine parameters.
- (*Headless mode*) Processes images with given settings, suitable for pipeline automation or after you've found good parameters.

### Configuration Optimization (*Meta-Learning*)

```python
from bookshelf_scanner import ConfigOptimizer

config_optimizer = ConfigOptimizer(extractor = extractor)
```

**Usage**:
```python
optimal_params = config_optimizer.optimize(image_files = image_files)
```

**What This Does**: 

- Instead of manually guessing parameters or relying on trial-and-error, this optimizer leverages a learned model.

- It performs multiple stochastic forward passes to estimate mean and variance of OCR performance, using them to choose new parameter sets that balance exploration (*trying new parameter regions*) and exploitation (*focusing on known good parameter areas*).
- Over several iterations, it converges to high-quality parameters with fewer evaluations than brute-force searching.

### Text Matching

```python
from fuzzy_matcher import FuzzyMatcher

matcher = FuzzyMatcher(
    min_match_score    = 0.8,
    max_matches        = 3,
    min_ocr_confidence = 0.1
)

matcher.match_books()
```

**What This Does**:  

- Takes the extracted text (*from OCR*) and queries a local DuckDB database of known book titles.

- Uses fuzzy string matching to accommodate OCR noise, partial strings, or different word orders.
- Outputs a ranked list of likely book matches for each extracted text snippet, aiding in automatically cataloging bookshelves.

---

## Additional Documentation

- [Spine Segmentation](./bookshelf_scanner/core/book_segmenter/README.md)

- [Text Extraction (TextExtractor)](./bookshelf_scanner/core/text_extractor/README.md)
- [Configuration Optimization (ConfigOptimizer)](./bookshelf_scanner/core/config_optimizer/README.md)
- [Fuzzy Matching (FuzzyMatcher)](./bookshelf_scanner/core/fuzzy_matcher/README.md)
- [Logging (ModuleLogger)](./bookshelf_scanner/core/module_logger/README.md)

---

## Configuration

- Default parameters are stored in `bookshelf_scanner/config/params.yml`.
- You can override parameters at runtime or through Python overrides, as shown earlier.
- Configurations determine which preprocessing steps are enabled, their ranges (*min, max, step*), and their default values.

---

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
