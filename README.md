# Bookshelf Scanner

## Table of Contents

- [Bookshelf Scanner](#bookshelf-scanner)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Running the Scanner in `parkington`](#running-the-scanner-in-parkington)
    - [Interactive Controls](#interactive-controls)
  - [Contributing](#contributing)

## Installation

This is the set of requirements I'm using for development, in case you're interesting in matching my patterns for local developments:

### Development Environment

- **MacOS**

  - **Homebrew:** Package manager for macOS. [Install Homebrew](https://brew.sh/)
  - **Tesseract OCR:** Installed via Homebrew
    ```bash
    brew install tesseract
    ```
- **Python:** Version 3.13
- **Poetry** Python package management installed via Homebrew — a TOML file has been supplied for quick shell installation, if interested

### Python Dependencies
Install the required Python dependencies using Poetry:

```bash
poetry install
```

## Usage

### Running the Scanner in `parkington`

1. **Prepare Image Directory**

   All the current bookshelf images are in the `images/` directory. The script is designed to pick up and use all of those files.

2. **Run the Scanner Script**

   Execute the `bookshelf_scanner.py` script using Poetry:

   ```bash
   poetry run python parkington/bookshelf_scanner.py
   ```

   The interactive interface will launch, displaying the first image from the `images/` directory.

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

## Contributing

Each contributor should create a unique directory (e.g., `yourlastname`) within the project to work on their features or improvements. Once your changes are ready, submit a pull request for review.