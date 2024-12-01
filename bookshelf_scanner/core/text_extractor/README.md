# TextExtractor

The `TextExtractor` module focuses on text extraction from book spine images, aided by an intuitive interactive interface. It combines multiple image processing techniques with OCR capabilities, allowing real-time parameter adjustment to handle challenging lighting conditions, varied text styles, and diverse spine layouts.

## Core Features

- **Interactive Parameter Tuning**
  
  - Real-time visual feedback for immediate assessment
  - Split-screen view showing original and processed images
  - Informative sidebar displaying current settings and controls

- **Image Processing Pipeline**
  
  - Shadow removal with adjustable kernel sizes for complex lighting
  - Color CLAHE (*Contrast Limited Adaptive Histogram Equalization*) for enhanced readability
  - Fine-grained brightness and contrast controls
  - 90-degree rotation adjustments for varied spine orientations
  - Each step can be independently toggled for optimal results

- **Advanced OCR Integration**
  
  - EasyOCR backend with GPU acceleration support
  - Confidence scoring for extracted text segments
  - Structured JSON output for downstream processing
  - Rotation-invariant text detection

## Usage

1. **Image Preparation**
   Place segmented book spine images in the `images/books` directory. The module accepts common image formats (JPG, PNG, BMP).

2. **Launch Interactive Interface**
   ```bash
   poetry run python TextExtractor/TextExtractor.py
   ```
   The interface presents your first image alongside an intuitive control panel.

### Interactive Controls

- **Processing Steps**
  
  - Number keys (`1`, `2`, `3`...) toggle individual processing steps
  - Each step's state (On/Off) is clearly displayed in the sidebar
  - Changes are applied instantly for immediate feedback

- **Parameter Adjustment**
  
  - Uppercase keys increase values (e.g., `B` for brightness)
  - Lowercase keys decrease values (e.g., `b` for brightness)
  - Current values are continuously updated in the sidebar
  - Parameters automatically scale to appropriate ranges

- **Navigation & Display**

  - `/`: Advance to next image
  - `q`: Gracefully exit, saving results

### Processing Pipeline Details

1. **Shadow Removal**
  
   - Eliminates uneven lighting and shadow artifacts
   - Adaptive kernel sizing for varying shadow intensities
   - Multi-channel processing preserves color information
   - Configurable median blur for noise reduction

2. **Color CLAHE**
  
   - Enhances local contrast while maintaining global balance
   - Adjustable clip limit prevents over-amplification
   - Particularly effective for faded or low-contrast text
   - Processes luminance channel to preserve color fidelity

3. **Brightness/Contrast**
  
   - Independent control over image luminance and dynamic range
   - Linear and non-linear adjustment options
   - Helps recover text from under/overexposed regions
   - Real-time histogram visualization

4. **Image Rotation**
  
   - 90-degree increment rotation
   - Maintains image quality through lossless transformation
   - Automatic dimension adjustment
   - Preserves aspect ratios

### Output Format

Results are saved to `ocr_results.json` in a structured format:
```json
{
    "image_name.jpg": [
        ["Extracted Text Segment", 0.95],
        ["Another Text Segment",   0.87],
        ["Series Information",     0.92]
    ]
}
```

## Programmatic Usage

The `TextExtractor` can be integrated into automated workflows:

```python
from TextExtractor import TextExtractor

# Initialize with GPU acceleration
extractor = TextExtractor(gpu_enabled = True)

# Configure and process images
results = extractor.interactive_experiment(
    image_files     = image_files,
    params_override = {
        'use_shadow_removal' : True,
        'shadow_kernel_size' : 23,
        'shadow_median_blur' : 15,
        'use_color_clahe'    : True,
        'clahe_clip_limit'   : 2.0
    },
    output_json    = True,
    interactive_ui = False
)
```