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

5. **Match Approval**:  
   After fuzzy matching, the `MatchApprover` module provides an interactive interface to review and confirm matches. Instead of trusting fuzzy matches blindly, you can interactively scroll through images, view their top candidate matches, and approve or skip each one. This ensures the final data used downstream (e.g., catalogs) is accurate and trustworthy.

**Logging with ModuleLogger**:  
Throughout, a `ModuleLogger` system provides consistent, module-specific logging. Each component records operations, parameter changes, and results, aiding in debugging, monitoring performance, and maintaining reproducibility.

---

## Project Structure

```
├── bookshelf_scanner/
│   ├── config/
│   │   └── params.yml        # Default parameter settings (*e.g., min/max for brightness*)
│   ├── core/
│   │   ├── book_segmenter/   # Spine detection code
│   │   ├── fuzzy_matcher/    # Fuzzy text matching
│   │   ├── match_approver/   # Interactive approval of matched titles
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

extractor   = TextExtractor()
image_files = extractor.find_image_files(subdirectory = 'books')
```

**Interactive Mode**:
```python
# Launch a UI window where you can toggle steps (1, 2, 3...) and adjust parameters (e.g., B/b for brightness)
extractor.run_interactive_mode(image_files = image_files)
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

### Match Approval (*Interactive*)

Once you have generated fuzzy match results (e.g., `matcher.json`), you can use the `MatchApprover` to review and finalize the matches. This step ensures that the final data entering your catalog or inventory system is both accurate and trusted.

**Command to Launch**:
```bash
poetry run match-approver
```

**What This Does**:  

- **Interactive UI**: Opens a simple, keyboard-driven interface displaying each image and its candidate matches side-by-side.

- **Approve with a Keypress**: Press a number key to immediately approve a corresponding match—no mouse clicks required.
- **Skip Non-Matching Images**: Press `s` if none of the listed matches fit; this ensures only correctly matched titles are recorded.
- **View Options**: Toggle between processed and raw image views (`/` key) to confirm that preprocessing steps haven’t distorted the text.
- **Easy Navigation**: Use `>` and `<` to move through the image list and `q` to quit once all approvals are done.

---

### Orchestrator (*Run the Entire Pipeline at Once*)

For users who want to run the entire pipeline (*or selected parts of it*) from a single command-line entry point, **Bookshelf Scanner** provides an orchestrator script. This script integrates all stages—book segmentation, parameter optimization, text matching, and match approval—into a single interface.

**Usage**:

By default, if you run the orchestrator without any flags, it will execute all steps in sequence:

```bash
poetry run bookshelf-scanner
```

This will:

1. **Book Segmenter**: Detect and segment out individual book spines.
2. **Config Optimizer**: Attempt to find optimal preprocessing steps and parameters for improved OCR results.
3. **Fuzzy Matcher**: Perform fuzzy matching of OCR-extracted text against a known database of book titles.
4. **Match Approver**: Launch an interactive interface to confirm or adjust matched titles.

If you prefer to run only certain parts of the pipeline, you can use the corresponding flags:

- `--book-segmenter`: Run only the BookSegmenter step.
- `--config-optimizer`: Run only the ConfigOptimizer step.
- `--fuzzy-matcher`: Run only the FuzzyMatcher step.
- `--match-approver`: Run only the MatchApprover step.

For example, to run just the segmentation and optimizer steps, you would run:

```bash
poetry run bookshelf-scanner --book-segmenter --config-optimizer
```

To specify a different directory for your input images:

```bash
poetry run bookshelf-scanner --images_dir images/custom_shelf
```

---

## Additional Documentation

- [Spine Segmentation](./bookshelf_scanner/core/book_segmenter/README.md)

- [Text Extraction (TextExtractor)](./bookshelf_scanner/core/text_extractor/README.md)
- [Configuration Optimization (ConfigOptimizer)](./bookshelf_scanner/core/config_optimizer/README.md)
- [Fuzzy Matching (FuzzyMatcher)](./bookshelf_scanner/core/fuzzy_matcher/README.md)
- [Match Approval (MatchApprover)](./bookshelf_scanner/core/match_approver/README.md)
- [Logging (ModuleLogger)](./bookshelf_scanner/core/module_logger/README.md)

---

## Future Work and Potential Enhancements

If we had more time to continue developing **Bookshelf Scanner**, we would focus on several key areas to streamline the workflow, improve scalability, and enhance robustness:

**1. Headless Orchestration of the Entire Pipeline**:  
Currently, each module (*e.g., spine segmentation, OCR preprocessing, configuration optimization*) operates somewhat independently and relies on reading and writing intermediate results to disk. In a future iteration, we would implement a fully headless orchestration layer that allows modules to pass data in-memory, avoiding unnecessary I/O overhead. This would create a more seamless, integrated pipeline where each stage can feed directly into the next without manual intervention or file-based communication.

**2. Parallelizing the Configuration Optimization Process**:  
The `ConfigOptimizer` currently evaluates each parameter set in sequence. By leveraging Python’s `multiprocessing` module, we could introduce parallel runs where multiple images or parameter sets are processed simultaneously. A manager process could distribute tasks to worker processes via a `Queue`, aggregate the results, and update the optimization model in real-time. This would greatly reduce the time needed to converge on optimal OCR parameters, especially for large libraries of images.

**3. Fallback Using Google Vision API**:  
For spine images that remain low-confidence after segmentation or OCR preprocessing, we could incorporate a fallback mechanism using the Google Vision API. Instead of applying all preprocessing techniques or repeatedly adjusting parameters, the system could selectively invoke Vision API calls for difficult cases. This pay-per-use approach would be more cost-effective than defaulting to external OCR services on every image, while still providing a reliable safety net for challenging spines.

**4. Expanding Pre-Processing Steps in `TextExtractor`**:  
While the current pipeline includes steps like shadow removal, CLAHE, and brightness/contrast adjustments, future refinements could incorporate more advanced preprocessing techniques. Potential additions include:

- **Binarization Strategies**: Adaptive thresholding, Otsu’s method, or Sauvola thresholding to isolate text foregrounds more reliably.
- **Noise Reduction & Deblurring**: Filters or neural network-based denoising and deblurring methods to tackle motion blur or grainy images.
- **Skew and Perspective Correction**: More sophisticated geometric transforms to correct not just rotation, but also perspective distortion and curvature.
- **Color Space Transformations**: Converting to alternative color spaces (e.g., HSV, LAB) for more robust contrast enhancements targeted at text regions.

**5. Improved Optimizer Trial Management**:  
Currently, the `ConfigOptimizer` runs a single trial of parameters per image and uses that information directly. A future iteration would allow the optimizer to run multiple parameter sets per image and automatically select the best-scoring configuration. By conducting multiple trials and comparing their performance, the pipeline could converge more reliably on optimal OCR parameters for each image, further enhancing the robustness and accuracy of the overall system.

---

## Configuration

- Default parameters are stored in `bookshelf_scanner/config/params.yml`.
- You can override parameters at runtime or through Python overrides, as shown earlier.
- Configurations determine which preprocessing steps are enabled, their ranges (*min, max, step*), and their default values.

---

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
