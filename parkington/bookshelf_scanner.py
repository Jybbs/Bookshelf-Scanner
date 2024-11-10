import cv2
import easyocr
import functools
import logging
import numpy as np

from dataclasses import dataclass, field
from pathlib     import Path
from ruamel.yaml import YAML
from typing      import Any

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
class Parameter:
    name         : str                       # Internal name of the parameter.
    display_name : str                       # Name to display in the UI.
    value        : Any                       # Current value of the parameter.
    min          : Any                       # Minimum value of the parameter.
    max          : Any                       # Maximum value of the parameter.
    step         : Any                       # Step size for incrementing/decrementing the parameter.
    increase_key : str                       # Key to increase the parameter.
    decrease_key : str = field(init = False) # Key to decrease the parameter (lowercase of increase_key).

    def __post_init__(self):
        self.decrease_key = self.increase_key.lower()

@dataclass
class ProcessingStep:
    name         : str                       # Internal name of the processing step.
    display_name : str                       # Name to display in the UI.
    toggle_key   : str                       # Key to toggle this processing step.
    parameters   : list[Parameter]           # List of parameter instances.
    is_enabled   : bool = False              # Whether the step is enabled (default: False).

    def toggle(self) -> str:
        """
        Toggle the 'is_enabled' state of the processing step and return the action message.
        """
        self.is_enabled = not self.is_enabled
        action_message  = f"Toggled '{self.display_name}' to {'On' if self.is_enabled else 'Off'}"
        return action_message

    def adjust_param(self, key_char: str) -> str:
        """
        Adjust the parameter value based on the provided key character and return the action message.
        """
        for param in self.parameters:
            if key_char == param.increase_key:
                old_value   = param.value
                param.value = min(param.value + param.step, param.max)
                action_message = f"Increased '{param.display_name}' from {old_value} to {param.value}"
                return action_message

            elif key_char == param.decrease_key:
                old_value   = param.value
                param.value = max(param.value - param.step, param.min)
                action_message = f"Decreased '{param.display_name}' from {old_value} to {param.value}"
                return action_message

        return None

# -------------------- Utility Functions --------------------

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

def ensure_odd(value: int) -> int:
    """
    Sets the least significant bit to 1, converting even numbers to the next odd number.
    """
    return value | 1

def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from the specified file path.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image

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
    """
    annotated       = base_image.copy()
    total_characters = 0
    
    if not params.get('use_show_annotations'):
        return annotated, total_characters

    sorted_contours = sorted(
        contours,
        key     = cv2.contourArea,
        reverse = True
    )
    
    # Process only the largest contours up to max_contours
    max_contours = min(int(params.get('max_contours', 10)), len(sorted_contours))
    for contour in sorted_contours[:max_contours]:
        cv2.drawContours(
            image      = annotated,
            contours   = [contour],
            contourIdx = -1,
            color      = (180, 0, 180),
            thickness  = 4
        )
        
        # Only perform OCR if enabled in parameters
        if perform_ocr and params.get('enable_ocr', False):
            # Get region from the processed image
            x, y, w, h = cv2.boundingRect(contour)
            ocr_region = ocr_image[y:y+h, x:x+w]
            
            if ocr_region.size > 0:
                # Convert to RGB for EasyOCR
                ocr_region_rgb = cv2.cvtColor(ocr_region, cv2.COLOR_BGR2RGB)
                
                # Perform OCR with book spine specific settings
                results = reader.readtext(
                    ocr_region_rgb,
                    decoder       = 'wordbeamsearch',
                    rotation_info = [90, 180, 270]
                )
                
                # Filter by confidence threshold
                min_confidence = params.get('ocr_confidence_threshold', 0.3)
                filtered_results = [res for res in results if res[2] >= min_confidence]
                
                if filtered_results:
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

def create_sidebar(
    steps           : list[ProcessingStep],
    sidebar_width   : int,
    current_display : str,
    image_name      : str,
    window_height   : int
) -> np.ndarray:
    """
    Creates a sidebar image displaying the controls and current settings.

    Args:
        steps           : List of processing steps.
        sidebar_width   : Width of the sidebar.
        current_display : Name of the current display option.
        image_name      : Name of the current image file.
        window_height   : Height of the window.

    Returns:
        Image of the sidebar.
    """
    sidebar        = np.zeros((window_height, sidebar_width, 3), dtype=np.uint8)
    scale_factor   = min(2.0, max(0.8, (window_height / 800) ** 0.5)) * 1.2
    font_scale     = 0.8 * scale_factor
    line_height    = int(32 * scale_factor)
    y_position     = int(30 * scale_factor)
    font_thickness = max(1, int(scale_factor * 1.5))

    def put_text(
        text  : str,
        x     : int,
        y     : int,
        color : tuple[int, int, int] = (255, 255, 255),
        scale : float = 1.0
    ):
        cv2.putText(
            img       = sidebar,
            text      = text,
            org       = (x, y),
            fontFace  = cv2.FONT_HERSHEY_DUPLEX,
            fontScale = font_scale * scale,
            color     = color,
            thickness = font_thickness,
            lineType  = cv2.LINE_AA
        )

    # Display current view option and image name
    put_text(
        text  = f"[/] View Options for {current_display}",
        x     = 10,
        y     = y_position,
        color = (255, 255, 0),
        scale = 1.1
    )
    y_position += line_height

    put_text(
        text  = f"   [?] Current Image: {image_name}",
        x     = 10,
        y     = y_position,
        color = (255, 255, 0),
        scale = 0.9
    )
    y_position += int(line_height * 1.5)

    # Display controls for each processing step
    for step in steps:
        put_text(
            text  = f"[{step.toggle_key}] {step.display_name}: {'On' if step.is_enabled else 'Off'}",
            x     = 10,
            y     = y_position,
            scale = 1.1
        )
        y_position += line_height

        for param in step.parameters:
            value_str = (
                f"{param.value:.3f}" if isinstance(param.value, float)
                else str(param.value)
            )
            key_text = f"[{param.decrease_key} | {param.increase_key}]"
            put_text(
                text  = f"   {key_text} {param.display_name}: {value_str}",
                x     = 20,
                y     = y_position,
                color = (200, 200, 200),
                scale = 0.9
            )
            y_position += line_height

        y_position += line_height

    # Display quit option
    put_text(
        text = "[q] Quit",
        x    = 10,
        y    = window_height - int(line_height * 1.5)
    )

    return sidebar

# -------------------- Main Interactive Function --------------------

def interactive_experiment(
    image_files     : list[Path],
    params_override : dict = None
):
    """
    Runs the interactive experiment allowing the user to adjust image processing parameters.

    Args:
        image_files     : List of image file paths to process.
        params_override : Optional; parameters to override default settings.

    Raises:
        ValueError: If no image files are provided.
    """
    if not image_files:
        raise ValueError("No image files provided")

    # Initialize variables and UI elements
    window_name         = 'Bookshelf Scanner'
    cv2.namedWindow('Bookshelf Scanner', cv2.WINDOW_NORMAL)
    current_image_idx   = 0
    display_options     = ['Original Image', 'Processed Image', 'Binary Image']
    current_display     = 0
    steps               = initialize_steps(params_override)
    original_image      = load_image(str(image_files[current_image_idx]))
    window_height       = max(original_image.shape[0], 800)
    scale_factor        = min(2.0, max(0.8, (window_height / 800) ** 0.5)) * 1.2
    sample_text         = "   [X | Y] Very Long Parameter Name: 100000.000"
    (width, _), _       = cv2.getTextSize(
        text      = sample_text,
        fontFace  = cv2.FONT_HERSHEY_DUPLEX,
        fontScale = 0.8 * scale_factor,
        thickness = 1
    )
    sidebar_width       = max(400, int(width * 1.2))
    last_params         = None
    cached_results      = None
    pending_log_message = None

    # Initialize EasyOCR Reader
    reader = easyocr.Reader(['en'], gpu=False)

    # Resize window to accommodate sidebar
    cv2.resizeWindow(
        window_name,
        original_image.shape[1] + sidebar_width,
        window_height
    )

    # Define key actions
    def quit_action():
        return 'quit'

    def toggle_display_action():
        return 'toggle_display'

    def next_image_action():
        return 'next_image'

    key_actions = {
        ord('q'): quit_action,
        ord('/'): toggle_display_action,
        ord('?'): next_image_action,
    }

    for step in steps:
        key_actions[ord(step.toggle_key)] = step.toggle
        for param in step.parameters:
            key_actions[ord(param.increase_key)] = functools.partial(step.adjust_param, param.increase_key)
            key_actions[ord(param.decrease_key)] = functools.partial(step.adjust_param, param.decrease_key)

    # Main loop
    while True:
        current_image_path = image_files[current_image_idx]

        if last_params is None:
            # Load new image and adjust window height
            original_image = load_image(str(current_image_path))
            window_height  = max(original_image.shape[0], 800)
            cv2.resizeWindow(
                window_name,
                original_image.shape[1] + sidebar_width,
                window_height
            )

        current_params = extract_params(steps)
        if last_params != current_params:

            # Process image and update cache
            processed_image, _, binary_image, contours = process_image(
                image = original_image,
                **current_params
            )
            last_params    = current_params.copy()
            cached_results = {
                'processed_image'     : processed_image,
                'binary_image'        : binary_image,
                'contours'            : contours,
                'annotated_original'  : None,
                'annotated_processed' : None,
                'annotated_binary'    : None,
                'total_characters'    : None,
                'num_contours'        : len(contours)
            }

            # Log results if we have a pending message
            if pending_log_message:
                log_message = pending_log_message
                if current_params.get('use_show_annotations'):
                    log_message += f" (Contours: {len(contours)}"
                    if current_params.get('enable_ocr'):
                        # Ensure we have character count
                        if cached_results['annotated_original'] is None:
                            _, total_characters = draw_annotations(
                                base_image  = original_image,
                                ocr_image   = processed_image,
                                contours    = contours,
                                params      = current_params,
                                reader      = reader,
                                perform_ocr = True
                            )
                            cached_results['total_characters'] = total_characters
                        log_message += f"; Characters: {cached_results['total_characters']}"
                    log_message += ")"
                logger.info(log_message)
                pending_log_message = None

        # Determine which base image to display
        if current_display == 0:
            display_image = original_image
            cache_key    = 'annotated_original'

        elif current_display == 1:
            display_image = cached_results['processed_image']
            cache_key    = 'annotated_processed'

        else:
            display_image = cached_results['binary_image']
            cache_key    = 'annotated_binary'

        # Apply annotations if enabled
        if current_params.get('use_show_annotations'):
            if cached_results[cache_key] is None:
                annotated_image, total_characters = draw_annotations(
                    base_image  = display_image,
                    ocr_image   = cached_results['processed_image'],
                    contours    = cached_results['contours'],
                    params      = current_params,
                    reader      = reader,
                    perform_ocr = True
                )
                cached_results[cache_key] = annotated_image
                if cache_key == 'annotated_original':
                    cached_results['total_characters'] = total_characters
            display_image = cached_results[cache_key]

        # Convert to BGR if grayscale
        if len(display_image.shape) == 2:
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)

        # Resize display image
        display_image = cv2.resize(
            src   = display_image,
            dsize = (
                int(display_image.shape[1] * (window_height / display_image.shape[0])),
                window_height
            )
        )

        # Create and show sidebar
        sidebar_image = create_sidebar(
            steps           = steps,
            sidebar_width   = sidebar_width,
            current_display = display_options[current_display],
            image_name      = current_image_path.name,
            window_height   = window_height
        )

        # Combine images and display
        main_display = np.hstack([display_image, sidebar_image])
        cv2.imshow(window_name, main_display)

        # Handle key events
        key = cv2.waitKey(1) & 0xFF
        action = key_actions.get(key)
        
        if action:
            action_result = action()

            if action_result == 'quit':
                break

            elif action_result == 'toggle_display':
                current_display = (current_display + 1) % len(display_options)
                logger.info(f"Switched to view: {display_options[current_display]}")

            elif action_result == 'next_image':
                current_image_idx = (current_image_idx + 1) % len(image_files)
                last_params      = None
                logger.info(f"Switched to image: {image_files[current_image_idx].name}")

            else:
                # Action result is an action message from toggle or adjust_param
                pending_log_message = action_result
                last_params = None
                if cached_results:
                    cached_results['annotated_original']  = None
                    cached_results['annotated_processed'] = None
                    cached_results['annotated_binary']    = None

        # Ensure log messages are written
        for handler in logger.handlers:
            handler.flush()

    cv2.destroyAllWindows()

# -------------------- Entry Point --------------------

if __name__ == "__main__":
    params_override = {
        'use_shadow_removal'      : True,
        'shadow_kernel_size'      : 11,
        'use_contour_adjustments' : True,
        'min_contour_area'        : 1000
    }

    image_files = get_image_files()
    interactive_experiment(image_files, params_override)
