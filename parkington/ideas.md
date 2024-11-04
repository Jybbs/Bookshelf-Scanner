### Contour Approximation
- Use `minAreaRect` to force contour shaping to always be a rectangle/quadrialteral, so that `RETR_EXTERNAL` can more easily eliminate internal contours from text
- Consider a `ProcessingStep` for `max_contours` for testing purposes

### Contour Filtering
- Remove the max `Parameter` and instead build in a permanent condition to not have a contour be larger than 90% of the image area

### OCR
- Ensure OCR only runs in regions surrounded by contours
- Ensure OCR only runs against the "Processed" image with Color CLAHE and Shadow Removal in play
- Does `pytesseract` have a `confidence` metric to optimize against? If so, design a parameter to only show highly-confident string retrievals