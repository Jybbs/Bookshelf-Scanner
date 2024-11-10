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
class DisplayOption:
    name      : str                         # Name of this display mode
    get_image : Callable[[], np.ndarray]    # Function to retrieve the display image
    cache_key : str                         # Key for accessing cached annotations

@dataclass
class DisplayState:
    image_idx     : int            = 0      # Index of current image in file list
    display_idx   : int            = 0      # Index of current display mode
    last_params   : Optional[dict] = None   # Previous processing parameters
    window_height : int            = 800    # Current window height in pixels
    
    def next_display(self) -> None:
        """
        Cycle to next display option.
        """
        self.display_idx = (self.display_idx + 1) % 3

    def next_image(self, total_images: int) -> None:
        """
        Advance to next image and reset processing state.
        """
        self.image_idx   = (self.image_idx + 1) % total_images
        self.last_params = None

@dataclass
class Parameter:
    name         : str                       # Internal name of the parameter.
    display_name : str                       # Name to display in the UI.
    value        : Any                       # Current value of the parameter.
    min          : Any                       # Minimum value of the parameter.
    max          : Any                       # Maximum value of the parameter.
    step         : Any                       # Step size for incrementing/decrementing the parameter.
    increase_key : str                       # Key to increase the parameter.
    decrease_key : str = field(init = False) # Key to decrease the parameter (lowercase of `increase_key``).

    def __post_init__(self):
        self.decrease_key = self.increase_key.lower()

@dataclass
class ProcessingStep:
    name         : str                       # Internal name of the processing step.
    display_name : str                       # Name to display in the UI.
    toggle_key   : str                       # Key to toggle this processing step.
    parameters   : list[Parameter]           # List of parameter instances.
    is_enabled   : bool = False              # Whether the step is enabled (default: False).

    def adjust_param(self, key_char: str) -> str:
        """
        Adjust the parameter value based on the provided key character and return the action message.
        """
        for param in self.parameters:

            if key_char in (param.increase_key, param.decrease_key):
                old_value      = param.value
                delta          = param.step if key_char == param.increase_key else -param.step
                param.value    = min(max(param.value + delta, param.min), param.max)
                action_message = f"{"Increased" if delta > 0 else "Decreased"} '{param.display_name}' from {old_value} to {param.value}"
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

def extract_params(steps: list[ProcessingStep]) -> dict:
    """
    Extract parameters from processing steps into a dictionary mapping names to values.
    """
    params = {
        param.name: param.value
        for step in steps
        for param in step.parameters
    }
    params.update({
        f"use_{step.name}": step.is_enabled
        for step in steps
    })
    return params

def get_image_files(images_dir: Path = None) -> list[Path]:
    """
    Retrieve a sorted list of image files from the nearest 'images' directory.
    """
    images_dir = images_dir or Path(__file__).resolve().parent

    for parent in [images_dir, *images_dir.parents]:
        potential_images_dir = parent / 'images'

        if potential_images_dir.is_dir():
            image_files = sorted(
                f for f in potential_images_dir.iterdir()
                if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
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
        with open(params_file, 'r') as f:
            step_definitions = yaml.load(f)

    except FileNotFoundError:
        logger.error(f"Configuration file not found: {params_file}")
        raise

    except Exception as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise

    steps = []
    for index, step_def in enumerate(step_definitions):
        toggle_key = str(index + 1)
        parameters = [Parameter(**param) for param in step_def.get('parameters', [])]

        steps.append(
            ProcessingStep(
                name         = step_def['name'],
                display_name = step_def['display_name'],
                toggle_key   = toggle_key,
                parameters   = parameters
            )
        )

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

def ocr_spine(
    spine_image : np.ndarray, 
    reader      : easyocr.Reader, 
    **params
) -> str:
    """
    Performs OCR on a given spine image using EasyOCR.

    Args:
        spine_image : The image of the book spine to perform OCR on.
        reader      : An instance of easyocr.Reader.
        **params    : Arbitrary keyword arguments containing OCR parameters.

    Returns:
        Extracted text from the spine image.
    """
    try:
        # EasyOCR expects RGB
        result = reader.readtext(
            cv2.cvtColor(spine_image, cv2.COLOR_BGR2RGB)
        )

        # Filter results based on confidence
        min_confidence = params.get('ocr_confidence_threshold', 0.3)
        text = ' '.join([res[1] for res in result if res[2] >= min_confidence])
        
        return text.strip()

    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return ''

def draw_annotations(
    base_image  : np.ndarray,
    ocr_image   : np.ndarray,
    contours    : list[np.ndarray],
    params      : dict,
    reader      : easyocr.Reader,
    perform_ocr : bool = True
) -> tuple[np.ndarray, int]:
    """
    Draws contours and recognized text on the image if annotations are enabled.
    
    Args:
        base_image  : Original image to annotate.
        ocr_image   : Image to use for OCR.
        contours    : List of contours to draw.
        params      : Dictionary of processing parameters.
        reader      : EasyOCR reader instance.
        perform_ocr : Whether to perform OCR even if enabled in params.
    
    Returns:
        tuple: (annotated image, total characters recognized)
    """
    if not params.get('use_show_annotations'):
        return base_image.copy(), 0
        
    annotated = base_image.copy()
    total_characters = 0
    
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:int(params.get('max_contours', 10))]
    
    for contour in sorted_contours:
        cv2.drawContours(
            image      = annotated,
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
            
        # Perform OCR with book spine specific settings
        results = reader.readtext(
            cv2.cvtColor(ocr_region, cv2.COLOR_BGR2RGB),
            decoder       = 'wordbeamsearch',
            rotation_info = [90, 180, 270]
        )
        
        # Filter by confidence threshold
        min_confidence = params.get('ocr_confidence_threshold', 0.3)
        filtered_results = [res for res in results if res[2] >= min_confidence]
        
        if not filtered_results:
            continue
            
        # Join all found text
        ocr_text = ' '.join(res[1] for res in filtered_results)
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
        
        # Draw background rectangle
        cv2.rectangle(
            img       = annotated,
            pt1       = (text_x - 5, text_y - text_height - 5),
            pt2       = (text_x + text_width + 5, text_y + 5),
            color     = (0, 0, 0),
            thickness = -1
        )
        
        # Draw text
        cv2.putText(
            img       = annotated,
            text      = ocr_text,
            org       = (text_x, text_y),
            fontFace  = cv2.FONT_HERSHEY_DUPLEX,
            fontScale = 0.6,
            color     = (255, 255, 255),
            thickness = 2
        )
        
    return annotated, total_characters

# -------------------- UI and Visualization Functions --------------------

def generate_sidebar_content(
    steps           : list[ProcessingStep],
    current_display : str,
    image_name      : str
) -> Iterator[tuple[str, tuple[int, int, int], float]]:
    """
    Generates the text content, colors, and scaling for each sidebar line.
    
    Args:
        steps           : List of processing steps to display.
        current_display : Name of current display mode.
        image_name      : Name of current image file.
        
    Returns:
        Iterator of tuples: (text content, RGB color, scale factor)
    """
    yield (f"[/] View Options for {current_display}", (255, 255, 0), 1.1)
    yield (f"   [?] Current Image: {image_name}", (255, 255, 0), 0.9)
    yield ("", (255, 255, 255), 1.0)  # Spacer
    
    for step in steps:
        yield (
            f"[{step.toggle_key}] {step.display_name}: {'On' if step.is_enabled else 'Off'}", 
            (255, 255, 255), 
            1.1
        )
        
        for param in step.parameters:
            value_str = f"{param.value:.3f}" if isinstance(param.value, float) else str(param.value)
            yield (
                f"   [{param.decrease_key} | {param.increase_key}] {param.display_name}: {value_str}",
                (200, 200, 200),
                0.9
            )
        
        yield ("", (255, 255, 255), 1.0)  # Spacer
    
    yield ("[q] Quit", (255, 255, 255), 1.0)

def render_sidebar_image(
    steps           : list[ProcessingStep],
    sidebar_width   : int,
    current_display : str,
    image_name      : str,
    window_height   : int
) -> np.ndarray:
    """
    Creates an image of the sidebar with all controls and settings.
    
    Args:
        steps           : List of processing steps to display.
        sidebar_width   : Width of the sidebar in pixels.
        current_display : Name of current display mode.
        image_name      : Name of current image file.
        window_height   : Height of the window in pixels.
    
    Returns:
        np.ndarray: Rendered sidebar image
    """
    sidebar        = np.zeros((window_height, sidebar_width, 3), dtype=np.uint8)
    scale_factor   = min(2.0, max(0.8, (window_height / 800) ** 0.5)) * 1.2
    font_scale     = 0.8 * scale_factor
    line_height    = int(32 * scale_factor)
    y_position     = int(30 * scale_factor)
    font_thickness = max(1, int(scale_factor * 1.5))

    for text, color, rel_scale in generate_sidebar_content(steps, current_display, image_name):
        if text:  # Skip empty lines
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
    state   = DisplayState()
    steps   = initialize_steps(params_override)
    reader  = easyocr.Reader(['en'], gpu = False)
    cache   = defaultdict(lambda: None)
    cache.update({'contours': [], 'num_contours': 0})

    # Configure display options
    def get_display_options(original: np.ndarray) -> list[DisplayOption]:
        return [
            DisplayOption('Original Image',  lambda: original,                 'annotated_original'),
            DisplayOption('Processed Image', lambda: cache['processed_image'], 'annotated_processed'),
            DisplayOption('Binary Image',    lambda: cache['binary_image'],    'annotated_binary')
        ]

    def setup_window(image: np.ndarray) -> tuple[str, int]:
        """
        Initialize window and calculate dimensions.
        """
        window_name         = 'Bookshelf Scanner'
        state.window_height = max(image.shape[0], 800)
        scale               = min(2.0, max(0.8, (state.window_height / 800) ** 0.5)) * 1.2
        
        sidebar_width = max(400, int(cv2.getTextSize(
            text      = "   [X | Y] Very Long Parameter Name: 100000.000",
            fontFace  = cv2.FONT_HERSHEY_DUPLEX,
            fontScale = 0.8 * scale,
            thickness = 1
        )[0][0] * 1.2))
        
        cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window_name, image.shape[1] + sidebar_width, state.window_height)
        return window_name, sidebar_width

    def update_processing_cache(image: np.ndarray):
        """
        Update cached processing results if parameters have changed.
        """
        current_params = extract_params(steps)

        if state.last_params != current_params:
            processed, _, binary, contours = process_image(image=image, **current_params)
            state.last_params = current_params.copy()
            cache.update({
                'processed_image'     : processed,
                'binary_image'        : binary,
                'contours'            : contours,
                'num_contours'        : len(contours),
                'annotated_original'  : None,
                'annotated_processed' : None,
                'annotated_binary'    : None,
                'total_characters'    : None
            })

    def handle_annotations(
        image       : np.ndarray, 
        params      : dict, 
        display_opt : DisplayOption
    ) -> np.ndarray:
        """
        Apply annotations to image if enabled.
        """
        if not params.get('use_show_annotations'):
            return image
            
        cache_key = display_opt.cache_key
        if cache[cache_key] is None:
            annotated, chars = draw_annotations(
                base_image  = image,
                ocr_image   = cache['processed_image'],
                contours    = cache['contours'],
                params      = params,
                reader      = reader,
                perform_ocr = True
            )
            cache[cache_key] = annotated
            cache['total_characters'] = chars
            
        return cache[cache_key]

    def handle_key_action(
            key          : int, 
            total_images : int
        ) -> Optional[str]:
            """
            Process a key press and return action message if any.
            """
            char = chr(key)
            
            if char == 'q':
                return 'quit'
            
            elif char == '/':
                state.next_display()
                return None
            
            elif char == '?':
                state.next_image(total_images)
                return None
                
            for step in steps:

                if char == step.toggle_key:
                    return step.toggle()
                
                if result := step.adjust_param(char):
                    return result
                
            return None

    # Main display loop
    original_image             = load_image(str(image_files[state.image_idx]))
    window_name, sidebar_width = setup_window(original_image)
    
    while True:
        # Refresh image if needed
        if state.last_params is None:
            original_image = load_image(str(image_files[state.image_idx]))
            cv2.resizeWindow(window_name, original_image.shape[1] + sidebar_width, state.window_height)
            
        # Update processing
        update_processing_cache(original_image)
        current_params = extract_params(steps)
        display_opt    = get_display_options(original_image)[state.display_idx]
        
        # Prepare display image
        display_image = display_opt.get_image()
        display_image = handle_annotations(display_image, current_params, display_opt)
        
        if len(display_image.shape) == 2:
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
            
        display_image = cv2.resize(
            src   = display_image,
            dsize = (int(display_image.shape[1] * (state.window_height / display_image.shape[0])), 
                    state.window_height)
        )

        # Show result
        cv2.imshow(window_name, np.hstack([
            display_image,
            render_sidebar_image(
                steps           = steps,
                sidebar_width   = sidebar_width,
                current_display = display_opt.name,
                image_name      = image_files[state.image_idx].name,
                window_height   = state.window_height
            )
        ]))

        # Handle input
        key = cv2.waitKey(1) & 0xFF
        if key == 255: # No key pressed
            continue
            
        if action := handle_key_action(key, len(image_files)):
            if action == 'quit':
                break
                
            logger.info(action)
            if current_params.get('use_show_annotations'):
                log_parts = [f"(Contours: {cache['num_contours']}"]
                if current_params.get('enable_ocr'):
                    log_parts.append(f"Characters: {cache['total_characters']}")
                logger.info(f"{action} {' '.join(log_parts)})")

            state.last_params = None
            for key in ['annotated_original', 'annotated_processed', 'annotated_binary']:
                cache[key] = None

    cv2.destroyAllWindows()

# -------------------- Entry Point --------------------

if __name__ == "__main__":

    image_files = get_image_files()
    interactive_experiment(image_files)
