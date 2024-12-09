# TextExtractor

The **TextExtractor** module provides a flexible interface for extracting text from book spine images using OCR, combined with a robust image processing pipeline. It includes both an **interactive mode**—where you can fine-tune processing steps and parameters in real-time—and a **headless mode**—for automated batch processing.

By adjusting steps such as shadow removal, contrast enhancement, and color normalization, `TextExtractor` prepares images to optimize OCR readability. This pipeline is versatile and supports a wide array of image conditions (e.g., uneven lighting, faded text, or unusual spine angles).

## Core Features

- **Interactive Parameter Adjustment**

  - Real-time feedback loop for immediate assessment of OCR improvements
  - Sidebar displaying current steps, parameter values, and keyboard controls
  - Key-based toggling of entire processing steps (e.g., shadow removal, CLAHE)
  - Incrementing/decrementing parameters (e.g., brightness or rotation angle) with single-keystroke commands

- **Image Processing Pipeline**

  Before running OCR, each image undergoes a series of transformations:
  
  1. **Shadow Removal**  
     Applies morphological operations and median blurs to even out uneven illumination.  
     This step normalizes illumination by dilating bright regions and subtracting shadows, ensuring text stands out more clearly.
  
  2. **Color CLAHE** (Contrast Limited Adaptive Histogram Equalization)  
     Improves local contrast without over-amplifying noise.  
     CLAHE is applied to the luminance channel in LAB color space, making faint text more distinguishable while preserving overall color balance.
  
  3. **Brightness and Contrast Adjustments**  
     Fine-tune illumination and dynamic range:  
     - **Brightness** shifts the intensity distribution upward or downward, making dim text more legible.  
     - **Contrast** scaling sharpens differences between text and background, crucial for extracting subtle lettering.
  
  4. **Image Rotation**  
     Rotates images by 90-degree increments to correct orientation issues.  
     This step ensures OCR sees upright text lines, improving recognition accuracy.

  Each of these steps can be enabled or disabled independently, and their parameters can be adjusted interactively. `TextExtractor` ensures these operations are applied consistently, allowing you to find the optimal preprocessing pipeline for your images.

- **OCR Integration**

  - Uses EasyOCR, which supports GPU acceleration, for efficient and reliable text detection.
  - Provides rotation-invariant OCR, attempting multiple rotations to find the best orientation for text reading.
  - Outputs confidence scores for detected text segments.
  - Structured results can be saved to JSON, facilitating downstream analysis or integration into other applications.

## Getting Started

### Image Preparation

Place your processed spine images in the `images/books` directory. Common formats like JPG, PNG, and BMP are supported.

### Running in Interactive Mode

Launch the interactive interface to experiment with processing steps and parameters:

```bash
poetry run text-extractor
```

You’ll see your first image and a control panel (sidebar). Parameters are displayed and updated in real-time as you press keys.

#### Interactive Controls

- **Step Toggling**:  
  Press digit keys (`1`, `2`, `3`, ...) to toggle steps like shadow removal or CLAHE on/off.
  
- **Parameter Adjustment**:  
  Each parameter has two associated keys, uppercase and lowercase.  
  For example, if `B` increases brightness, then `b` decreases it. Pressing these keys updates the parameter instantly, allowing you to see the effect on OCR outputs right away.

- **Navigation**:
  - `/` cycles to the next image in the directory.
  - `q` quits the interface and closes the window.

### Processing Pipeline Details

1. **Shadow Removal**  
   Employs morphological dilation and median blurring to estimate and remove background gradients.  
   By equalizing illumination, text edges become clearer and less obscured by shadows.

2. **Color CLAHE**  
   Uses a localized form of histogram equalization limited by a `clip_limit` to avoid over-contrast.  
   This makes subtle text more visible, especially in cases of faded ink or low local contrast.

3. **Brightness/Contrast**  
   Adjusts the histogram of the image:  
   - **Brightness** shifts all pixel intensities up or down, helpful for images that are too dark or too bright.  
   - **Contrast** modifies the range of intensities, increasing or decreasing the difference between text and background.

4. **Rotation**  
   By aligning text orientation, OCR can more accurately segment and recognize characters. Since `TextExtractor` tries multiple angles, this mitigates the issue of tilted or sideways spines.

### Output Format

After processing, OCR results can be saved as JSON (if enabled):

```json
{
  "image_name.jpg": [
    ["Extracted Text Segment", 0.95],
    ["Another Text Segment",   0.87]
  ]
}
```

This output is suitable for further analysis, record-keeping, or integration with other modules.

## Usage

### Basic Usage in Python

```python
from bookshelf_scanner import TextExtractor

extractor = TextExtractor(headless = False)

image_files = extractor.find_image_files(subdirectory = 'Books')
extractor.run_interactive_mode(image_files = image_files)
```

In `run_interactive_mode`, you can freely tweak parameters until you achieve satisfactory OCR results.

### Headless Mode

If you prefer non-interactive batch processing:

```python
from pathlib import Path
from bookshelf_scanner import TextExtractor

extractor = TextExtractor(
    headless     = True,
    output_json  = True,
    output_file  = Path('custom_output.json')
)

image_files = extractor.find_image_files(subdirectory = 'Books')
extractor.run_headless_mode(image_files = image_files)
```

This processes images without opening a window or requiring user input, saving the OCR results directly to a JSON file.

### Configuration Overrides

You can pass a `config_override` dictionary to modify specific parameters programmatically before starting:

```python
config_override = {
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
    },
    'brightness_adjustment' : {
        'enabled'    : True,
        'parameters' : {
            'brightness_value' : 10
        }
    }
}

extractor.initialize_processing_steps(config_override = config_override)
```

This ensures your custom settings are applied consistently across runs.

### Further Optimization with ConfigOptimizer

For automated configuration optimization without manual tuning, consider using our [ConfigOptimizer](../config_optimizer/README.md). It employs a meta-learning approach and Bayesian-inspired acquisition strategies to find effective parameter settings efficiently, saving you from manual trial-and-error.