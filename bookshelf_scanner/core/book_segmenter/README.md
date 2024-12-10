Below is a revised README that reflects the updated code structure and functionality of the `BookSegmenter` module:

---

# BookSegmenter

The **BookSegmenter** module leverages a locally stored [OpenShelves (version 8)](https://universe.roboflow.com/capjamesg/open-shelves/model/8) YOLO-based model to detect and segment books from images of bookshelves. This model was trained on the [OpenShelves dataset](https://universe.roboflow.com/capjamesg/open-shelves) and can accurately identify the bounding boxes and masks of books without requiring an internet connection or API keys.

By running inference directly from the included ONNX model (`OpenShelves8.onnx`), the `BookSegmenter` ensures reproducible and stable results, unaffected by changes to external APIs or model hosting platforms.

## Features

- **Standalone Inference**:  
  No API keys or external connections required. The ONNX model is stored locally.
  
- **Accurate Book Detection and Segmentation**:  
  Identify books in an image and produce bounding boxes, confidence scores, and segmentation masks.  
  Masks isolate individual books, enabling further processing like OCR, classification, or metadata extraction.
  
- **Optional Output**:  
  - **Cropped Book Images**: Save segmented book images for inspection or downstream tasks.
  - **JSON Results**: Store detection metadata (bounding boxes, confidence scores) as structured JSON data.

## Running via Script

From the project root, you can run a quick demo to verify that the `BookSegmenter` is working correctly on your machine:

```bash
poetry run book-segmenter
```

This command will run the module's default functionality (*you may need to modify it if the script requires specific arguments or configuration*).

## Usage

To use the `BookSegmenter` in your own code, ensure that your current working directory is the project root, or that your `PYTHONPATH` is set appropriately. You can then import and use the `BookSegmenter` class as follows:

```python
from bookshelf_scanner import BookSegmenter
from pathlib           import Path

# Initialize the BookSegmenter with default thresholds
segmenter = BookSegmenter(
    confidence_thresh = 0.3,
    iou_thresh        = 0.5,
    output_images     = True,  # Save cropped book images
    output_json       = True   # Save results as JSON
)

# Path to the bookshelf image you want to process
image_path = Path("path/to/bookshelf_image.jpg")

# Segment books
results = segmenter.segment_books(image_path)

# `results` is a dictionary like:
# {
#   "books": [
#       {
#           "file_name": "bookshelf_image_001.jpg",
#           "confidence": 0.95,
#           "bounding_box": [x1, y1, x2, y2]
#       },
#       ...
#   ]
# }

print(results)
```

### Direct Image Segmentation

If you prefer finer control, you can call lower-level methods:

```python
import cv2
from bookshelf_scanner import BookSegmenter

segmenter = BookSegmenter(output_images = False, output_json = False)
image = cv2.imread("path/to/bookshelf_image.jpg")

# Directly get segments, bounding boxes, and confidences
segments, bboxes, confidences = segmenter.segment_image(image, use_masks = True)

for i, (segment, bbox, conf) in enumerate(zip(segments, bboxes, confidences)):
    print(f"Book {i+1}: Confidence = {conf}, BBox = {bbox}")
    # `segment` is a cropped and masked image array of the detected book
```

---

## Future Plans

- [ ] Further optimization to speed up inference and segmentation.
- [ ] Evaluate alternative models or integrate custom YOLO variants for improved accuracy.
- [ ] Add batching support for segmenting multiple images at once.
- [ ] Enhance mask post-processing for cleaner book extractions.