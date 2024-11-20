1. **Prepare Image Directory**

   All the current segmented book images are in the `images/books` directory. The script is designed to pick up and use all of those files.

2. **Run the Scanner Script**

   Execute the `TextExractor.py` script using Poetry:

   ```bash
  poetry run python TextExtractor/TextExtractor.py
   ```

   The interactive interface will launch, displaying the first image from the `images/books` directory.

### Interactive Controls

The scanner provides an interactive UI to adjust processing parameters in real-time. Below are the key controls:

- **Toggle Processing Steps**
  - Press the corresponding number key (`1`, `2`, `3`, etc.) to enable or disable a processing step.
  
- **Adjust Parameters**
  - For each parameter within a processing step:
    - Press the uppercase key (e.g., `K`, `B`, `G`) to increase the parameter value.
    - Press the lowercase key (e.g., `k`, `b`, `g`) to decrease the parameter value.
  
- **View Options**
  - Press `/` to cycle through different display options:
    - **Processed Image:** Shows the image after processing steps.
    - **Binary Image:** Displays the binary thresholded image.
    - **Annotated Image:** Shows contours and OCR results overlaid on the original image.
  
- **Navigate Images**
  - Press `?` to switch to the next image in the `images/` directory.
  
- **Quit Application**
  - Press `q` to exit the scanner.

**Sidebar Overview:**

The sidebar displays the current settings and available controls:

- **Processing Steps:** Each step can be toggled on/off.
- **Parameters:** Adjustable parameters for each processing step with current values.
- **Current View:** Indicates which display option is active.
- **Image Information:** Shows the name of the current image being processed.