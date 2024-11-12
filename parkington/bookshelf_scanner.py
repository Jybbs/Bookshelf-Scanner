import cv2
import easyocr
import logging
import numpy as np

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib     import Path
from ruamel.yaml import YAML
from typing      import Any, Callable, Iterator, Optional

# -------------------- Configuration and Logging --------------------

logging.basicConfig(
    level    = logging.INFO,
    format   = '%(asctime)s - %(levelname)s - %(message)s',
    filename = 'bookshelf_scanner.log',
    filemode = 'w'
)
logger = logging.getLogger(__name__)

# -------------------- Data Classes --------------------

@dataclass
class DisplayState:
    image_idx       : int                        = 0     # Index of current image
    display_idx     : int                        = 0     # Index of current display mode
    window_height   : int                        = 800   # Window height in pixels
    last_params     : Optional[dict]             = None  # Previous processing parameters
    annotations     : dict[str, np.ndarray]      = field(default_factory = dict)  # Cached annotations
    original_image  : Optional[np.ndarray]       = None  # The original image
    processed_image : Optional[np.ndarray]       = None  # The processed image
    binary_image    : Optional[np.ndarray]       = None  # The binary (thresholded) image
    contours        : Optional[list[np.ndarray]] = None  # Contours found in the image
    image_name      : str                        = ''    # Name of the current image

    def next_display(self, total_displays: int):
        """
        Cycle to the next display option.
        """
        self.display_idx = (self.display_idx + 1) % total_displays

    def next_image(self, total_images: int):
        """
        Cycle to the next image and reset image-related state.
        """
        self.image_idx += 1
        self.image_idx %= total_images
        self.original_image = None

    def reset_image_state(self):
        """
        Resets processing-related state variables, keeping the original image and image name.
        """
        self.processed_image = None
        self.binary_image    = None
        self.contours        = None
        self.annotations     = {}
        self.last_params     = None

@dataclass
class Parameter:
    name         : str  # Internal name of the parameter.
    display_name : str  # Name to display in the UI.
    value        : Any  # Current value of the parameter.
    min          : Any  # Minimum value of the parameter.
    max          : Any  # Maximum value of the parameter.
    step         : Any  # Step size for incrementing/decrementing the parameter.
    increase_key : str  # Key to increase the parameter.

    @property
    def decrease_key(self) -> str:
        """
        The `decrease_key` is always the lowercase of the `increase_key`.
        """
        return self.increase_key.lower()

@dataclass
class ProcessingStep:
    name         : str  # Internal name of the processing step.
    display_name : str  # Name to display in the UI.
    toggle_key   : str  # Key to toggle this processing step.
    parameters   : list[Parameter]  # List of parameter instances.
    is_enabled   : bool = False     # Whether the step is enabled (default: False).

    def adjust_param(self, key_char: str) -> str:
        """
        Adjust the parameter value based on the provided key character and return the action message.
        """
        for param in self.parameters:

            if key_char in (param.increase_key, param.decrease_key):
                old_value      = param.value
                delta          = param.step if key_char == param.increase_key else -param.step
                param.value    = min(max(param.value + delta, param.min), param.max)
                action_message = f"{'Increased' if delta > 0 else 'Decreased'} '{param.display_name}' from {old_value} to {param.value}"
                return action_message
        
        return None
    
    def toggle(self) -> str:
        """
        Toggle the 'is_enabled' state of the processing step and return the action message.
        """
        self.is_enabled = not self.is_enabled
        action_message  = f"Toggled '{self.display_name}' to {'On' if self.is_enabled else 'Off'}"
        return action_message

# -------------------- Utility Functions --------------------

def ensure_odd(value: int) -> int:
    """
    Sets the least significant bit to 1, converting even numbers to the next odd number.
    """
    return value | 1

def extract_params(steps: list[ProcessingStep]) -> dict[str, Any]:
    """
    Extract parameters from processing steps into a dictionary mapping names to values.
    """
    return {
        **{param.name: param.value for step in steps for param in step.parameters},
        **{f"use_{step.name}": step.is_enabled for step in steps}
    }

def get_image_files(images_dir: Optional[Path] = None) -> list[Path]:
    """
    Retrieve a sorted list of image files from the nearest 'images' directory.
    """
    images_dir = images_dir or Path(__file__).resolve().parent
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    for parent in [images_dir, *images_dir.parents]:
        potential_images_dir = parent / 'images'

        if potential_images_dir.is_dir():
            image_files = sorted(
                f for f in potential_images_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            )
            if image_files:
                return image_files

    raise FileNotFoundError("No image files found in an 'images' directory.")

def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from the specified file path.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image

# -------------------- Initialization Function --------------------

def initialize_steps(params_override: dict = None) -> list[ProcessingStep]:
    """
    Initializes processing steps with default parameters or overrides.

    Args:
        params_override : Optional; a dictionary of parameters to override default settings.

    Returns:
        A list of initialized ProcessingStep instances.

    Raises:
        FileNotFoundError : If the configuration file is not found.
        Exception         : If there is an error parsing the configuration file.
    """
    params_file = Path(__file__).resolve().parent / 'params.yml'
    yaml        = YAML(typ = 'safe')

    try:
        with params_file.open('r') as f:
            step_definitions = yaml.load(f)

    except FileNotFoundError:
        logger.error(f"Configuration file not found: {params_file}")
        raise

    except Exception as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise

    steps = [
        ProcessingStep(
            name         = step_def['name'],
            display_name = step_def['display_name'],
            toggle_key   = str(index + 1),
            parameters   = [Parameter(**param) for param in step_def.get('parameters', [])]
        )
        for index, step_def in enumerate(step_definitions)
]

    if params_override:
        step_map  = {f"use_{step.name}": step for step in steps}
        param_map = {param.name: param for step in steps for param in step.parameters}

        for key, value in params_override.items():
            if key in step_map:
                step_map[key].is_enabled = value
            elif key in param_map:
                param_map[key].value = value

    return steps

# -------------------- Image Processing Functions --------------------

def process_image(
    image : np.ndarray,
    **params
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    Processes the image according to the parameters provided.

    Args:
        image    : Original image to process.
        **params : Arbitrary keyword arguments containing processing parameters.

    Returns:
        A tuple containing:
            - Processed color image.
            - Grayscale image.
            - Binary image.
            - List of contours.
    """
    processed = image.copy()

    # Shadow Removal
    if params.get('use_shadow_removal'):
        k_size    = ensure_odd(int(params['shadow_kernel_size']))
        blur_size = ensure_odd(int(params['shadow_median_blur']))
        kernel    = np.ones((k_size, k_size), np.uint8)
        channels  = list(cv2.split(processed))  # Convert tuple to list for modification

        for i in range(len(channels)):
            dilated     = cv2.dilate(channels[i], kernel)
            background  = cv2.medianBlur(dilated, blur_size)
            difference  = 255 - cv2.absdiff(channels[i], background)
            channels[i] = cv2.normalize(difference, None, 0, 255, cv2.NORM_MINMAX)

        processed = cv2.merge(channels)

    # Bilateral Filter
    if params.get('use_bilateral_filter'):
        diameter    = int(params['bilateral_diameter'])
        sigma_color = params['bilateral_sigma_color']
        sigma_space = params['bilateral_sigma_space']
        processed   = cv2.bilateralFilter(processed, diameter, sigma_color, sigma_space)

    # Color CLAHE
    if params.get('use_color_clahe'):
        lab          = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        clahe        = cv2.createCLAHE(clipLimit=params['clahe_clip_limit'])
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        processed    = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Convert to Grayscale
    grayscale = 255 - cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    # Edge Detection
    if params.get('use_edge_detection'):
        edges     = cv2.Canny(grayscale, params['canny_threshold1'], params['canny_threshold2'])
        grayscale = cv2.bitwise_or(grayscale, edges)

    # Adaptive Thresholding
    if params.get('use_adaptive_thresholding'):
        b_size = ensure_odd(int(params['adaptive_block_size']))
        binary = cv2.adaptiveThreshold(
            src            = grayscale,
            maxValue       = 255,
            adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType  = cv2.THRESH_BINARY_INV,
            blockSize      = b_size,
            C              = params['adaptive_c']
        )
    else:
        _, binary = cv2.threshold(
            src    = grayscale,
            thresh = 0,
            maxval = 255,
            type   = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    # Find Contours
    contours, _ = cv2.findContours(
        image   = binary,
        mode    = cv2.RETR_EXTERNAL,
        method  = cv2.CHAIN_APPROX_SIMPLE
    )

    # Contour Adjustments
    if params.get('use_contour_adjustments'):
        min_area   = params['min_contour_area']
        image_area = image.shape[0] * image.shape[1]
        contours   = [
            contour for contour in contours
            if min_area <= cv2.contourArea(contour) <= 0.9 * image_area
        ]

        max_contours = int(params['max_contours'])
        contours     = contours[:max_contours]

        # Contour Approximation
        if params['contour_approximation']:
            contours = [
                cv2.boxPoints(cv2.minAreaRect(contour)).astype(int)
                for contour in contours
            ]

    return processed, grayscale, binary, contours

# -------------------- Image Annotation Functions --------------------

def extract_text_from_image(
    image  : np.ndarray,
    reader : easyocr.Reader,
    **params
) -> str:
    """
    Extracts text from a given image using EasyOCR.

    Args:
        image    : The image to perform OCR on.
        reader   : An instance of easyocr.Reader.
        **params : Arbitrary keyword arguments containing OCR parameters.

    Returns:
        Extracted text from the image.
    """
    try:
        min_confidence = params.get('ocr_confidence_threshold', 0.3)
        result = reader.readtext(
            image[..., ::-1], # BGR to RGB conversion
            decoder       = 'wordbeamsearch',
            rotation_info = params.get('ocr_rotation_info', [90, 180, 270])
        )

        text = ' '.join(res[1] for res in result if res[2] >= min_confidence).strip()
        return text
    
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return ''

def annotate_image_with_text(
    original_image : np.ndarray,
    ocr_image      : np.ndarray,
    contours       : list[np.ndarray],
    params         : dict,
    reader         : easyocr.Reader,
    perform_ocr    : bool = True
) -> tuple[np.ndarray, int]:
    """
    Draws contours and recognized text on the image if annotations are enabled.

    Args:
        original_image : Original image to annotate.
        ocr_image      : Image to use for OCR.
        contours       : List of contours to draw.
        params         : Dictionary of processing parameters.
        reader         : EasyOCR reader instance.
        perform_ocr    : Whether to perform OCR even if enabled in params.

    Returns:
        Tuple containing the annotated image and total characters recognized.
    """
    if not params.get('use_show_annotations'):
        return original_image.copy(), 0

    annotated_image  = original_image.copy()
    total_characters = 0
    max_contours     = int(params.get('max_contours', 10))

    sorted_contours = sorted(
        contours,
        key     = cv2.contourArea,
        reverse = True
    )[:max_contours]

    for contour in sorted_contours:
        cv2.drawContours(
            image      = annotated_image,
            contours   = [contour],
            contourIdx = -1,
            color      = (180, 0, 180),
            thickness  = 4
        )

        if not (perform_ocr and params.get('enable_ocr')):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        ocr_region = ocr_image[y:y+h, x:x+w]

        if ocr_region.size == 0:
            continue

        ocr_text = extract_text_from_image(
            image  = ocr_region,
            reader = reader,
            **params
        )

        if not ocr_text:
            continue

        total_characters += len(ocr_text)

        # Center text in contour
        (text_width, text_height), _ = cv2.getTextSize(
            text      = ocr_text,
            fontFace  = cv2.FONT_HERSHEY_DUPLEX,
            fontScale = 0.6,
            thickness = 2
        )
        text_x = x + (w - text_width) // 2
        text_y = y + (h + text_height) // 2

        # Draw background rectangle and text
        cv2.rectangle(
            img       = annotated_image,
            pt1       = (text_x - 5, text_y - text_height - 5),
            pt2       = (text_x + text_width + 5, text_y + 5),
            color     = (0, 0, 0),
            thickness = -1
        )
        cv2.putText(
            img       = annotated_image,
            text      = ocr_text,
            org       = (text_x, text_y),
            fontFace  = cv2.FONT_HERSHEY_DUPLEX,
            fontScale = 0.6,
            color     = (255, 255, 255),
            thickness = 2
        )

    return annotated_image, total_characters

# -------------------- UI and Visualization Functions --------------------

def generate_sidebar_content(
    steps        : list[ProcessingStep],
    display_name : str,
    image_name   : str
) -> list[tuple[str, tuple[int, int, int], float]]:
    """
    Generates a list of sidebar lines with text content, colors, and scaling factors.
    
    Args:
        steps        : List of processing steps to display.
        display_name : Name of current display mode.
        image_name   : Name of current image file.
        
    Returns:
        List of tuples: (text content, RGB color, scale factor)
    """
    TEAL  = (255, 255, 0)
    WHITE = (255, 255, 255)
    GRAY  = (200, 200, 200)
    
    lines = [
        (f"[/] View Options for {display_name}", TEAL, 1.1),
        (f"   [?] Current Image: {image_name}",  TEAL, 0.9),
        ("", WHITE, 1.0)  # Spacer
    ]
    
    for step in steps:
        status = 'On' if step.is_enabled else 'Off'
        lines.append((f"[{step.toggle_key}] {step.display_name}: {status}", WHITE, 1.1))

        for param in step.parameters:
            value_str = f"{param.value:.3f}" if isinstance(param.value, float) else str(param.value)
            lines.append((
                f"   [{param.decrease_key} | {param.increase_key}] {param.display_name}: {value_str}",
                GRAY,
                0.9
            ))
        lines.append(("", WHITE, 1.0))  # Spacer
    
    lines.append(("[q] Quit", WHITE, 1.0))
    return lines

def render_sidebar(
    steps         : list[ProcessingStep],
    sidebar_width : int,
    display_name  : str,
    image_name    : str,
    window_height : int
) -> np.ndarray:
    """
    Renders the sidebar image with controls and settings.
    
    Args:
        steps         : List of processing steps to display.
        sidebar_width : Width of the sidebar in pixels.
        display_name  : Name of current display mode.
        image_name    : Name of current image file.
        window_height : Height of the window in pixels.
    
    Returns:
        np.ndarray: Rendered sidebar image
    """
    sidebar        = np.zeros((window_height, sidebar_width, 3), dtype=np.uint8)
    scale          = min(2.0, max(0.8, (window_height / 800) ** 0.5)) * 1.2
    font_scale     = 0.8 * scale
    line_height    = int(32 * scale)
    y_position     = int(30 * scale)
    font_thickness = max(1, int(scale * 1.5))
    
    for text, color, rel_scale in generate_sidebar_content(steps, display_name, image_name):
        if text:
            cv2.putText(
                img       = sidebar,
                text      = text,
                org       = (10, y_position),
                fontFace  = cv2.FONT_HERSHEY_DUPLEX,
                fontScale = font_scale * rel_scale,
                color     = color,
                thickness = font_thickness,
                lineType  = cv2.LINE_AA
            )
        y_position += line_height
    
    return sidebar

# -------------------- Main Interactive Function --------------------

def interactive_experiment(
    image_files     : list[Path],
    params_override : dict = None
):
    """
    Runs the interactive experiment allowing parameter adjustment and image processing.
    
    Args:
        image_files     : List of image file paths to process.
        params_override : Optional parameter overrides.
    """
    if not image_files:
        raise ValueError("No image files provided")

    # Initialize components
    state  = DisplayState()
    steps  = initialize_steps(params_override)
    reader = easyocr.Reader(['en'], gpu=False)

    # Display options
    display_options = [
        ('Original Image',  lambda: state.original_image,  'annotated_original'),
        ('Processed Image', lambda: state.processed_image, 'annotated_processed'),
        ('Binary Image',    lambda: state.binary_image,    'annotated_binary')
    ]
    total_displays = len(display_options)

    # Initialize window
    window_name = 'Bookshelf Scanner'
    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)

    while True:
        if state.original_image is None:  # Load image only if not already loaded
            image_path           = image_files[state.image_idx]
            state.original_image = load_image(str(image_path))
            state.window_height  = max(state.original_image.shape[0], 800)
            state.image_name     = image_path.name
            state.reset_image_state()

        # Extract current parameters
        current_params = extract_params(steps)

        # Process image if parameters have changed
        if state.last_params != current_params:
            state.processed_image, _, state.binary_image, state.contours = process_image(state.original_image, **current_params)
            state.last_params = current_params.copy()
            state.annotations = {}

        # Get current display image
        display_name, get_image_func, cache_key = display_options[state.display_idx]
        display_image = get_image_func()

        # Apply annotations if enabled
        if current_params.get('use_show_annotations'):
            if cache_key not in state.annotations:
                annotated_image, _ = annotate_image_with_text(
                    original_image = display_image,
                    ocr_image      = state.processed_image,
                    contours       = state.contours,
                    params         = current_params,
                    reader         = reader
                )
                state.annotations[cache_key] = annotated_image
            display_image = state.annotations[cache_key]

        # Prepare display image
        if display_image.ndim == 2:
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)

        display_scale = state.window_height / display_image.shape[0]
        display_image = cv2.resize(
            display_image,
            (int(display_image.shape[1] * display_scale), state.window_height)
        )

        # Render sidebar
        sidebar_width = 1400
        sidebar_image = render_sidebar(
            steps         = steps,
            sidebar_width = sidebar_width,
            display_name  = display_name,
            image_name    = state.image_name,
            window_height = state.window_height
        )

        # Combine images and display
        combined_image = np.hstack([display_image, sidebar_image])
        cv2.imshow(window_name, combined_image)
        cv2.resizeWindow(window_name, combined_image.shape[1], combined_image.shape[0])

        # Handle key input
        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            continue  # No key pressed

        char = chr(key)
        if char == 'q':
            break

        elif char == '/':
            state.next_display(total_displays)

        elif char == '?':
            state.next_image(len(image_files))

        else:
            param_changed = False
            for step in steps:
                if char == step.toggle_key:
                    logger.info(step.toggle())
                    param_changed = True
                    break

                for param in step.parameters:
                    if char in (param.increase_key, param.decrease_key):
                        action = step.adjust_param(char)
                        if action:
                            logger.info(action)
                            param_changed = True
                            break

                if param_changed:
                    break

            if param_changed:
                state.last_params = None  # Force reprocessing when parameters change

    cv2.destroyAllWindows()

# -------------------- Entry Point --------------------

if __name__ == "__main__":

    image_files = get_image_files()
    interactive_experiment(image_files)
