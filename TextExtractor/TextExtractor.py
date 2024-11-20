import cv2
import easyocr
import logging
import numpy as np

from dataclasses import dataclass, field
from pathlib     import Path
from ruamel.yaml import YAML
from typing      import Any, Optional

# -------------------- Configuration and Logging --------------------

logging.basicConfig(
    level    = logging.INFO,
    format   = '%(asctime)s - %(levelname)s - %(message)s',
    filename = 'TextExtractor.log',
    filemode = 'w'
)
logger = logging.getLogger(__name__)

# -------------------- Data Classes --------------------

@dataclass
class DisplayState:
    image_idx       : int                   = 0     # Index of current image
    display_idx     : int                   = 0     # Index of current display mode
    window_height   : int                   = 800   # Window height in pixels
    last_params     : Optional[dict]        = None  # Previous processing parameters
    annotations     : dict[str, np.ndarray] = field(default_factory = dict)  # Cached annotations
    original_image  : Optional[np.ndarray]  = None  # The original image
    processed_image : Optional[np.ndarray]  = None  # The processed image
    binary_image    : Optional[np.ndarray]  = None  # The binary (thresholded) image
    image_name      : str                   = ''    # Name of the current image

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
        self.annotations     = {}
        self.last_params     = None

@dataclass
class Parameter:
    name          : str         # Internal name of the parameter.
    display_name  : str         # Name to display in the UI.
    value         : Any         # Current value of the parameter.
    increase_key  : str         # Key to increase the parameter.
    min           : Any = None  # Minimum value of the parameter.
    max           : Any = None  # Maximum value of the parameter.
    step          : Any = None  # Step size for incrementing/decrementing the parameter.

    @property
    def decrease_key(self) -> str:
        """
        The `decrease_key` is always the lowercase of the `increase_key`.
        """
        return self.increase_key.lower()
    
    @property
    def display_value(self) -> str:
        """
        Returns the value formatted as a string for display purposes.
        """
        if isinstance(self.value, float):
            return f"{self.value:.2f}"
        else:
            return str(self.value)

    def __post_init__(self):
        if self.min is None or self.max is None or self.step is None:
            raise ValueError(f"Parameter '{self.name}' must have 'min', 'max', and 'step' defined.")

    def adjust_value(self, increase: bool):
        """
        Adjusts the parameter value based on whether the parameter should be increased or decreased.
        """
        old_value = self.value
        delta     = self.step if increase else -self.step
        new_value = self.value + delta
        if isinstance(self.value, float):
            self.value = round(max(self.min, min(new_value, self.max)), 2)
        else:
            self.value = max(self.min, min(new_value, self.max))
        return old_value  # Return old value for logging purposes

@dataclass
class ProcessingStep:
    name          : str              # Internal name of the processing step.
    display_name  : str              # Name to display in the UI.
    toggle_key    : str              # Key to toggle this processing step.
    parameters    : list[Parameter]  # List of parameter instances.
    is_enabled    : bool = False     # Whether the step is enabled (default: False).

    def adjust_param(self, key_char: str) -> Optional[str]:
        """
        Adjust the parameter value based on the provided key character and return the action message.
        """
        for param in self.parameters:
            if key_char in (param.increase_key, param.decrease_key):
                increase      = key_char == param.increase_key
                old_value     = param.adjust_value(increase)
                old_value_str = f"{old_value:.2f}" if isinstance(old_value, float) else str(old_value)
                new_value_str = f"{param.value:.2f}" if isinstance(param.value, float) else str(param.value)
                action_type   = 'Increased' if increase else 'Decreased'
                return f"{action_type} '{param.display_name}' from {old_value_str} to {new_value_str}"
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

def find_image_files(target_subdirectory: str = 'images/books', start_directory: Optional[Path] = None) -> list[Path]:
    """
    Retrieve a sorted list of image files from the specified target_subdirectory.
    """
    start_directory = start_directory or Path(__file__).resolve().parent
    allowed_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    image_files = next(
        (
            sorted(
                file for file in (directory / target_subdirectory).rglob('*')
                if file.is_file() and file.suffix.lower() in allowed_image_extensions
            )
            for directory in [start_directory, *start_directory.parents]
            if (directory / target_subdirectory).is_dir()
        ),
        None
    )

    if image_files:
        return image_files

    raise FileNotFoundError(f"No image files found in '{target_subdirectory}' directory.")

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
            parameters   = [Parameter(**param_def) for param_def in step_def.get('parameters', [])]
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Processes the image according to the parameters provided.

    Args:
        image    : Original image to process.
        **params : Arbitrary keyword arguments containing processing parameters.

    Returns:
        A tuple containing:
            - Processed color image.
            - Binary image.
    """
    processed_image = image.copy()

    # Brightness Adjustment
    if params.get('use_brightness_adjustment'):
        brightness_value   = params['brightness_value']
        hsv_image          = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 2] = cv2.add(hsv_image[:, :, 2], brightness_value)
        processed_image    = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Contrast Adjustment
    if params.get('use_contrast_adjustment'):
        contrast_value  = params['contrast_value']
        processed_image = cv2.convertScaleAbs(processed_image, alpha=contrast_value, beta=0)

    # Shadow Removal
    if params.get('use_shadow_removal'):
        shadow_kernel_size = ensure_odd(int(params['shadow_kernel_size']))
        shadow_median_blur = ensure_odd(int(params['shadow_median_blur']))
        shadow_kernel      = np.ones((shadow_kernel_size, shadow_kernel_size), np.uint8)
        channels           = list(cv2.split(processed_image))  # Convert tuple to list for modification

        for i in range(len(channels)):
            dilated_image    = cv2.dilate(channels[i], shadow_kernel)
            background_image = cv2.medianBlur(dilated_image, shadow_median_blur)
            difference_image = 255 - cv2.absdiff(channels[i], background_image)
            channels[i]      = cv2.normalize(difference_image, None, 0, 255, cv2.NORM_MINMAX)

        processed_image = cv2.merge(channels)

    # Gaussian Blur
    if params.get('use_gaussian_blur'):
        gaussian_kernel_size = ensure_odd(int(params['gaussian_kernel_size']))
        processed_image = cv2.GaussianBlur(
            src    = processed_image, 
            ksize  = (gaussian_kernel_size, gaussian_kernel_size), 
            sigmaX = params['gaussian_sigma']
        )

    # Color CLAHE
    if params.get('use_color_clahe'):
        lab_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2LAB)
        clahe     = cv2.createCLAHE(clipLimit=params['clahe_clip_limit'])
        lab_image[:, :, 0] = clahe.apply(lab_image[:, :, 0])
        processed_image    = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    # Convert to Grayscale
    grayscale_image = 255 - cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

    # Adaptive Thresholding
    if params.get('use_adaptive_thresholding'):
        adaptive_block_size = ensure_odd(int(params['adaptive_block_size']))
        binary_image = cv2.adaptiveThreshold(
            src            = grayscale_image,
            maxValue       = 255,
            adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType  = cv2.THRESH_BINARY_INV,
            blockSize      = adaptive_block_size,
            C              = params['adaptive_c']
        )
    else:
        _, binary_image = cv2.threshold(
            src    = grayscale_image,
            thresh = 0,
            maxval = 255,
            type   = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    return processed_image, binary_image

# -------------------- Image Annotation Functions --------------------

def extract_text_from_image(
    image  : np.ndarray,
    reader : easyocr.Reader,
    **params
) -> list:
    """
    Extracts text from a given image using EasyOCR.

    Args:
        image    : The image to perform OCR on.
        reader   : An instance of easyocr.Reader.
        **params : Arbitrary keyword arguments containing OCR parameters.

    Returns:
        List of tuples containing OCR results.
    """
    try:
        min_confidence = params.get('ocr_confidence_threshold', 0.3)
        ocr_results = reader.readtext(
            image[..., ::-1],  # BGR to RGB conversion
            decoder       = 'greedy',
            rotation_info = [90, 180, 270]
        )

        # Filter results by confidence
        ocr_results = [result for result in ocr_results if result[2] >= min_confidence]

        return ocr_results

    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return []

def draw_rotated_text(
    image      : np.ndarray,
    text       : str,
    position   : tuple[int, int],
    angle      : float,
    font_scale : float                = 0.3,
    color      : tuple[int, int, int] = (0, 0, 255),
    thickness  : int                  = 1
) -> np.ndarray:
    """
    Draws rotated text on an image at a specified position and angle.

    Args:
        image      : The image to draw on.
        text       : The text string to draw.
        position   : (x, y) tuple for the position to place the text.
        angle      : Angle in degrees to rotate the text.
        font_scale : Font scale for the text.
        color      : Color of the text in BGR.
        thickness  : Thickness of the text strokes.

    Returns:
        The image with the text drawn on it.
    """
    font                        = cv2.FONT_HERSHEY_SIMPLEX
    text_size, baseline         = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height     = text_size
    text_image                  = np.zeros((text_height + baseline, text_width, 3), dtype=np.uint8)

    # Draw outline text
    cv2.putText(
        img       = text_image,
        text      = text,
        org       = (0, text_height),
        fontFace  = font,
        fontScale = font_scale,
        color     = (255, 255, 255),
        thickness = thickness + 2,
        lineType  = cv2.LINE_AA
    )
    # Draw main text
    cv2.putText(
        img       = text_image,
        text      = text,
        org       = (0, text_height),
        fontFace  = font,
        fontScale = font_scale,
        color     = color,
        thickness = thickness,
        lineType  = cv2.LINE_AA
    )

    # Rotate the text image
    center                 = (text_width / 2, (text_height + baseline) / 2)
    rotation_matrix        = cv2.getRotationMatrix2D(center, angle, 1.0)
    abs_cosine             = abs(rotation_matrix[0, 0])
    abs_sine               = abs(rotation_matrix[0, 1])
    bounding_width         = int((text_height + baseline) * abs_sine + text_width * abs_cosine)
    bounding_height        = int((text_height + baseline) * abs_cosine + text_width * abs_sine)
    rotation_matrix[0, 2] += bounding_width / 2 - center[0]
    rotation_matrix[1, 2] += bounding_height / 2 - center[1]

    rotated_text_image = cv2.warpAffine(
        src         = text_image,
        M           = rotation_matrix,
        dsize       = (bounding_width, bounding_height),
        flags       = cv2.INTER_LINEAR,
        borderMode  = cv2.BORDER_CONSTANT,
        borderValue = (0, 0, 0)
    )

    # Create mask and extract region of interest
    rotated_gray_image = cv2.cvtColor(rotated_text_image, cv2.COLOR_BGR2GRAY)
    _, mask            = cv2.threshold(rotated_gray_image, 1, 255, cv2.THRESH_BINARY)
    height, width      = rotated_text_image.shape[:2]
    x_pos, y_pos       = position
    x_pos             -= width // 2
    y_pos             -= height // 2
    x_pos              = max(0, min(x_pos, image.shape[1] - width))
    y_pos              = max(0, min(y_pos, image.shape[0] - height))
    region_of_interest = image[y_pos:y_pos + height, x_pos:x_pos + width]

    # Ensure the region of interest is in BGR format and correct size
    if region_of_interest.ndim == 2 or region_of_interest.shape[2] == 1:
        region_of_interest = cv2.cvtColor(region_of_interest, cv2.COLOR_GRAY2BGR)
    if region_of_interest.shape[:2] != (height, width):
        region_of_interest = cv2.resize(region_of_interest, (width, height))

    # Combine the text image with the region of interest
    image_background = cv2.bitwise_and(region_of_interest, region_of_interest, mask = cv2.bitwise_not(mask))
    text_foreground  = cv2.bitwise_and(rotated_text_image, rotated_text_image, mask = mask)
    combined_image   = cv2.add(image_background, text_foreground)
    image[y_pos:y_pos + height, x_pos:x_pos + width] = combined_image

    return image

def annotate_image_with_text(
    original_image : np.ndarray,
    ocr_image      : np.ndarray,
    params         : dict,
    reader         : easyocr.Reader
) -> np.ndarray:
    """
    Annotates the image with recognized text.

    Args:
        original_image : Original image to annotate.
        ocr_image      : Image to use for OCR.
        params         : Dictionary of processing parameters.
        reader         : EasyOCR reader instance.

    Returns:
        Annotated image.
    """
    # Ensure the annotated image is in BGR format
    if original_image.ndim == 2 or original_image.shape[2] == 1:
        annotated_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        annotated_image = original_image.copy()

    ocr_results = extract_text_from_image(image=ocr_image, reader=reader, **params)

    for bounding_box, text, confidence in ocr_results:
        coordinates = np.array(bounding_box).astype(int)
        # Draw bounding box around detected text
        cv2.polylines(
            img       = annotated_image,
            pts       = [coordinates],
            isClosed  = True,
            color     = (0, 255, 0),
            thickness = 2
        )
        logger.info(f"OCR Text: '{text}' with confidence {confidence:.2f}")

        x_coordinates, y_coordinates = coordinates[:, 0], coordinates[:, 1]
        top_edge_length      = np.hypot(x_coordinates[1] - x_coordinates[0], y_coordinates[1] - y_coordinates[0])
        right_edge_length    = np.hypot(x_coordinates[2] - x_coordinates[1], y_coordinates[2] - y_coordinates[1])
        text_with_confidence = f"{text} ({confidence * 100:.1f}%)"

        if right_edge_length > top_edge_length:
            # If the box is taller than it is wide, place text to the right and rotate it
            angle          = 270
            max_x          = np.max(x_coordinates)
            min_y, max_y   = np.min(y_coordinates), np.max(y_coordinates)
            text_position  = (int(max_x + 10), int((min_y + max_y) / 2))
        else:
            # If the box is wider than it is tall, place text above it
            angle          = 0
            min_x, max_x   = np.min(x_coordinates), np.max(x_coordinates)
            min_y          = np.min(y_coordinates)
            text_position  = (int((min_x + max_x) / 2), int(min_y - 10))

        annotated_image = draw_rotated_text(
            image    = annotated_image,
            text     = text_with_confidence,
            position = text_position,
            angle    = angle
        )

    return annotated_image

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
        (f"[/] View Options for {display_name}", TEAL, 1.0),
        (f"   [?] Current Image: {image_name}",  TEAL, 0.85),
        ("", WHITE, 1.0)  # Spacer
    ]
    
    for step in steps:
        status = 'On' if step.is_enabled else 'Off'
        lines.append((f"[{step.toggle_key}] {step.display_name}: {status}", WHITE, 1.0))

        for param in step.parameters:
            value_str = param.display_value
            lines.append((
                f"   [{param.decrease_key} | {param.increase_key}] {param.display_name}: {value_str}",
                GRAY,
                0.85
            ))
        lines.append(("", WHITE, 1.0))  # Spacer
    
    lines.append(("[q] Quit", WHITE, 1.0))
    return lines

def render_sidebar(
    steps         : list[ProcessingStep],
    display_name  : str,
    image_name    : str,
    window_height : int
) -> np.ndarray:
    """
    Renders the sidebar image with controls and settings.
    
    Args:
        steps         : List of processing steps to display.
        display_name  : Name of current display mode.
        image_name    : Name of current image file.
        window_height : Height of the window in pixels.
    
    Returns:
        np.ndarray: Rendered sidebar image
    """
    lines          = generate_sidebar_content(steps, display_name, image_name)
    num_lines      = len(lines)
    margin         = int(0.05 * window_height)
    line_height    = max(20, min(int((window_height - 2 * margin) / num_lines), 40))
    font_scale     = line_height / 40
    font_thickness = max(1, int(font_scale * 1.5))
    font           = cv2.FONT_HERSHEY_DUPLEX

    # Determine maximum text width
    max_text_width = max(
        cv2.getTextSize(text, font, font_scale * rel_scale, font_thickness)[0][0]
        for text, _, rel_scale in lines if text
    )

    # Sidebar width with padding
    sidebar_width = max_text_width + 20  # 10 pixels padding on each side

    # Create sidebar image
    sidebar = np.zeros((window_height, sidebar_width, 3), dtype=np.uint8)

    # Draw text lines
    y_position = margin + line_height
    for text, color, rel_scale in lines:
        if text:
            cv2.putText(
                img       = sidebar,
                text      = text,
                org       = (10, y_position),
                fontFace  = font,
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
    reader = easyocr.Reader(['en'], gpu = False)

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
            state.processed_image, state.binary_image = process_image(state.original_image, **current_params)
            state.last_params = current_params.copy()
            state.annotations = {}

        # Get current display image
        display_name, get_image_func, cache_key = display_options[state.display_idx]
        display_image = get_image_func()

        # Apply annotations if OCR is enabled
        if current_params.get('use_ocr'):
            if cache_key not in state.annotations:
                annotated_image = annotate_image_with_text(
                    original_image = display_image,
                    ocr_image      = state.processed_image,
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
        sidebar_image = render_sidebar(
            steps         = steps,
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
            for step in steps:
                if char == step.toggle_key:
                    logger.info(step.toggle())
                    state.last_params = None  # Force reprocessing
                    break

                for param in step.parameters:
                    if char in (param.increase_key, param.decrease_key):
                        action = step.adjust_param(char)
                        if action:
                            logger.info(action)
                            state.last_params = None  # Force reprocessing
                            break
                else:
                    continue
                break

    cv2.destroyAllWindows()

# -------------------- Entry Point --------------------

if __name__ == "__main__":

    image_files = find_image_files('images/books')
    interactive_experiment(image_files)
