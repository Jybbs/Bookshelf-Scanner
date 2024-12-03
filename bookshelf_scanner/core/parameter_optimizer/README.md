# ParameterOptimizer

The `ParameterOptimizer` module implements a parameter optimization overlay for the `TextExtractor` pipeline. It systematically explores the parameter space to identify optimal processing settings for each image, using a character-weighted confidence scoring system to evaluate OCR effectiveness.

## Overview

The optimizer employs the following exploration strategy:

1. Analyzes enabled processing steps from TextExtractor

2. Generates valid parameter combinations respecting step dependencies
3. Evaluates OCR results using multi-factor scoring
4. Maintains best parameters per image with detailed metrics
5. Provides robust progress tracking and recovery capabilities

## Key Features

- **Parameter Space Exploration**

  - Respects processing step dependencies
  - Generates only valid parameter combinations
  - Adapts to enabled/disabled processing steps
  - Efficient batch processing for memory management

- **Sophisticated Scoring System**

  - Character-count weighted confidence scoring
  - Optional character-type weighting capabilities
  - Balanced evaluation of length vs. confidence

- **Progress Management**

  - Periodic result saving with configurable frequency
  - Resume capability for interrupted operations
  - Detailed progress logging
  - Memory-efficient batch processing

## Getting Started

1. **Image Preparation**  
   Place your test images in the `images/books` directory. The optimizer accepts common image formats (JPG, PNG, BMP).

2. **Run the Optimizer**  
   ```bash
   poetry run parameter-optimizer
   ```
   The optimizer will process your images and display progress in its dedicated log. Results will be saved automatically, including the optimal parameters discovered and detailed performance metrics.

## Implementation

```python
from ParameterOptimizer import ParameterOptimizer
from TextExtractor      import TextExtractor

# Initialize with GPU support
extractor = TextExtractor(gpu_enabled = True)
optimizer = ParameterOptimizer(
    extractor      = extractor,
    output_file    = 'optimized_results.json',
    batch_size     = 100,
    save_frequency = 10
)

# Run optimization with resume capability
results = optimizer.optimize(
    image_files = spine_images,
    resume      = True
)
```

### Configuration

Parameter ranges are defined in `params.yml`:
```yaml
- name: shadow_removal
  display_name: Shadow Removal
  parameters:
    - name         : shadow_kernel_size
      display_name : Shadow Kernel Size
      value        : 23
      min          : 1
      max          : 45
      step         : 2
```

### Detailed Output Format

The optimizer produces comprehensive JSON results:
```json
{
    "image_name.jpg": {
        "best_parameters": {
            "shadow_kernel_size": 23,
            "clahe_clip_limit": 2.0,
            "brightness_value": 10,
            "contrast_value": 1.2,
            "use_shadow_removal": true,
            "use_color_clahe": true
        },
        "text_results": [
            ["Detailed Text Sample", 0.95],
            ["Additional Text", 0.87]
        ],
        "metrics": {
            "score": 156.8,
            "char_count": 185,
            "iterations": 42
        }
    }
}
```