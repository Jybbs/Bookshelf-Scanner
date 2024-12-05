import cv2
import json
import numpy as np

from dataclasses   import dataclass
from easyocr       import Reader
from functools     import cache
from pathlib       import Path
from PIL           import Image, ImageDraw, ImageFont
from ruamel.yaml   import YAML
from typing        import Any

from bookshelf_scanner import ModuleLogger, Utils
logger = ModuleLogger('extractor')()

# -------------------- Data Classes --------------------

@dataclass
class DisplayState:
    """
    Tracks the current state of image processing and display.
    """
    image_idx     : int = 0      # Index of the current image
    image_name    : str = ''     # Name of the current image
    window_height : int = 800    # Window height in pixels

    def next_image(self, total_images: int) -> None:
        """
        Cycle to the next image.
        """
        self.image_idx = (self.image_idx + 1) % total_images

@dataclass
class Parameter:
    """
    Defines an adjustable parameter for image processing.
    """
    name         : str                          # Internal name of the parameter
    display_name : str                          # Name to display in the UI
    value        : int | float                  # Current value of the parameter
    increase_key : str                          # Key to increase the parameter
    min          : int | float | None = None    # Minimum value of the parameter
    max          : int | float | None = None    # Maximum value of the parameter
    step         : int | float | None = None    # Step size for incrementing/decrementing the parameter

    @property
    def decrease_key(self) -> str:
        """
        Returns the key to decrease the parameter.
        """
        return self.increase_key.lower()

    @property
    def display_value(self) -> str:
        """
        Returns the value formatted as a string for display.
        """
        return f"{self.value:.2f}" if isinstance(self.value, float) else str(self.value)

    def adjust_value(self, increase: bool) -> int | float:
        """
        Adjusts the parameter value based on direction.

        Args:
            increase : True to increase the value, False to decrease

        Returns:
            Previous value before adjustment
        """
        old_value = self.value
        delta     = self.step if increase else -self.step
        new_value = self.value + delta

        if isinstance(self.value, float):
            self.value = round(max(self.min, min(new_value, self.max)), 2)
        else:
            self.value = max(self.min, min(new_value, self.max))

        return old_value

@dataclass
class ProcessingStep:
    """
    Groups related parameters and processing logic.
    """
    name         : str                # Internal name of the processing step
    display_name : str                # Name to display in the UI
    toggle_key   : str                # Key to toggle this processing step
    parameters   : list[Parameter]    # List of parameter instances
    is_enabled   : bool = False       # Whether the step is enabled

    def adjust_param(self, key_char: str) -> str | None:
        """
        Adjusts parameter value based on key press.

        Args:
            key_char : The character representing the key pressed

        Returns:
            Action description string or None if no action taken
        """
        for param in self.parameters:
            if key_char in (param.increase_key, param.decrease_key):
                increase    = key_char == param.increase_key
                old_value   = param.adjust_value(increase)
                action_type = 'Increased' if increase else 'Decreased'
                return f"{action_type} '{param.display_name}' from {old_value} to {param.value}"
        return None

    def toggle(self) -> str:
        """
        Toggles the enabled state of the processing step.

        Returns:
            Description of the action taken
        """
        self.is_enabled = not self.is_enabled
        return f"'{self.display_name}' {('Enabled' if self.is_enabled else 'Disabled')}"

# -------------------- ProcessingState Class --------------------

@dataclass(frozen = True)
class ProcessingState:
    """
    Immutable state representing all processing parameters for an image.
    Used as a cache key for consistent image processing results.
    """
    steps: tuple[tuple[str, bool, tuple[Any, ...]], ...]

    @classmethod
    def from_steps(cls, steps: list[ProcessingStep]) -> 'ProcessingState':
        """
        Creates an immutable ProcessingState from a list of ProcessingSteps.
        Only includes enabled steps and their parameters, with consistent ordering
        for reliable caching.

        Args:
            steps: List of ProcessingStep instances to convert

        Returns:
            ProcessingState: Immutable state instance containing enabled steps and their parameters
        """
        state_steps = tuple(
            (
                step.name,
                step.is_enabled,
                tuple(sorted(
                    (param.name, param.value)
                    for param in sorted(step.parameters, key = lambda p: p.name)
                ))
            )
            for step in sorted(steps, key = lambda s: s.name)
            if step.is_enabled
        )
        return cls(steps = state_steps)

    def to_current_parameters(self) -> dict:
        """
        Converts the ProcessingState to the current_parameters dictionary format.

        Returns:
            Dictionary of nested parameters organized by processing step
        """
        return {
            step_name: {
                'enabled'    : step_enabled,
                'parameters' : dict(step_params)
            }
            for step_name, step_enabled, step_params in self.steps
        }

# -------------------- Processing Functions --------------------

def adjust_brightness(image: np.ndarray, params: dict) -> np.ndarray:
    """
    Adjusts the brightness of the image, which can enhance the visibility of text in
    underexposed or overexposed images, improving OCR accuracy.
    """
    value     = params['brightness_value']
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 2] = cv2.add(hsv_image[:, :, 2], value)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def adjust_contrast(image: np.ndarray, params: dict) -> np.ndarray:
    """
    Adjusts the contrast of the image to enhance the distinction between text and background.
    This is beneficial for OCR when the text is not clearly separated from the background.
    """
    alpha = params['contrast_value']
    return cv2.convertScaleAbs(image, alpha = alpha, beta = 0)

def apply_clahe(image: np.ndarray, params: dict) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the image, enhancing
    local contrast and revealing text details in varying lighting conditions, which aids OCR.
    """
    clip_limit = params['clahe_clip_limit']
    lab_image  = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe      = cv2.createCLAHE(clipLimit = clip_limit)

    lab_image[:, :, 0] = clahe.apply(lab_image[:, :, 0])
    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

def ocr_processing(image: np.ndarray, params: dict) -> np.ndarray:
    """
    Placeholder function for OCR processing step.
    This function does not modify the image but ensures that 'ocr' is treated as a valid processing step.
    """
    return image

def remove_shadow(image: np.ndarray, params: dict) -> np.ndarray:
    """
    Removes shadows from the image, which is essential for OCR when shadows obscure parts
    of the text, making it difficult for the OCR engine to recognize characters correctly.
    """
    kernel_size = int(params['shadow_kernel_size']) | 1
    median_blur = int(params['shadow_median_blur']) | 1
    kernel      = np.ones((kernel_size, kernel_size), np.uint8)
    channels    = list(cv2.split(image))

    for i in range(len(channels)):
        dilated     = cv2.dilate(channels[i], kernel)
        bg_image    = cv2.medianBlur(dilated, median_blur)
        diff_image  = 255 - cv2.absdiff(channels[i], bg_image)
        channels[i] = cv2.normalize(diff_image, None, 0, 255, cv2.NORM_MINMAX)

    return cv2.merge(channels)

def rotate_image(image: np.ndarray, params: dict) -> np.ndarray:
    """
    Rotates the image by a specified angle, which is useful for correcting the orientation
    of the text in images, ensuring that the OCR engine can read it accurately.
    """
    angle = params['rotation_angle']
    if angle % 360 != 0:
        k = int(angle / 90) % 4
        return np.rot90(image, k)
    return image

PROCESSING_FUNCTIONS = {
    'brightness_adjustment' : adjust_brightness,
    'color_clahe'           : apply_clahe,
    'contrast_adjustment'   : adjust_contrast,
    'image_rotation'        : rotate_image,
    'ocr'                   : ocr_processing,
    'shadow_removal'        : remove_shadow,
}

# -------------------- TextExtractor Class --------------------

class TextExtractor:
    PROJECT_ROOT    = Utils.find_root('pyproject.toml')
    ALLOWED_FORMATS = {'.bmp', '.jpg', '.jpeg', '.png', '.tiff'}
    DEFAULT_HEIGHT  = 800
    FONT_FACE       = cv2.FONT_HERSHEY_DUPLEX
    OUTPUT_FILE     = PROJECT_ROOT / 'bookshelf_scanner' / 'data' / 'results' / 'extractor.json'
    PARAMS_FILE     = PROJECT_ROOT / 'bookshelf_scanner' / 'config' / 'params.yml'
    UI_COLORS       = {
        'GRAY'  : (200, 200, 200),
        'TEAL'  : (255, 255, 0),
        'WHITE' : (255, 255, 255)
    }
    WINDOW_NAME     = 'Bookshelf Scanner'

    def __init__(
        self,
        allowed_formats : set[str] | None = None,
        gpu_enabled     : bool            = False,
        headless        : bool            = False,
        output_json     : bool            = False,
        output_file     : Path | None     = None,
        params_file     : Path | None     = None,
        window_height   : int             = DEFAULT_HEIGHT,
    ):
        """
        Initializes the TextExtractor instance.

        Args:
            allowed_formats : Set of allowed image extensions (defaults to ALLOWED_FORMATS)
            gpu_enabled     : Whether to use GPU for OCR processing
            headless        : Whether to run in headless mode
            output_json     : Whether to output OCR results to JSON file
            output_file     : Path to save resultant strings from OCR processing
            params_file     : Optional custom path to params.yml
            window_height   : Default window height for UI display
        """
        self.allowed_formats = allowed_formats or self.ALLOWED_FORMATS
        self.headless        = headless
        self.output_json     = output_json
        self.output_file     = output_file or self.OUTPUT_FILE
        self.params_file     = params_file or self.PARAMS_FILE
        self.reader          = Reader(['en'], gpu = gpu_enabled)
        self.steps           = []  # Will be populated by initialize_steps

        if not self.headless:
            self.state = DisplayState(window_height = window_height)
        else:
            self.state = None

    @property
    def current_parameters(self) -> dict:
        """
        Returns the current set of parameters across all processing steps.

        Returns:
            Dictionary of nested parameters organized by processing step
        """
        processing_state = ProcessingState.from_steps(self.steps)
        return processing_state.to_current_parameters()

    @classmethod
    def find_image_files(cls, subdirectory: str = 'Books') -> list[Path]:
        """
        Retrieves a sorted list of image files from a subdirectory within the images folder.

        Args:
            subdirectory : Name of the subdirectory within 'images' to process (e.g., 'Books', 'Bookcases', 'Shelves')

        Returns:
            List of image file paths

        Raises:
            FileNotFoundError : If no image files are found in the specified subdirectory
        """
        image_dir = cls.PROJECT_ROOT / 'bookshelf_scanner' / 'images' / subdirectory
        
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image subdirectory not found: {image_dir}")
            
        image_files = sorted(
            file for file in image_dir.glob('*')
            if file.is_file() and file.suffix.lower() in cls.ALLOWED_FORMATS
        )
        
        if not image_files:
            raise FileNotFoundError(f"No image files found in {image_dir}")
            
        return image_files

    @staticmethod
    def center_image_in_square(image: np.ndarray) -> np.ndarray:
        """
        Centers the image in a square canvas with sides equal to the longest side.
        """
        height, width = image.shape[:2]
        max_side      = max(height, width)
        canvas        = np.zeros((max_side, max_side, 3), dtype = np.uint8)
        y_offset      = (max_side - height) // 2
        x_offset      = (max_side - width)  // 2

        canvas[y_offset:y_offset + height, x_offset:x_offset + width] = image
        return canvas

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """
        Loads an image from the specified file path.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return image

    def initialize_steps(self, params_override: dict | None = None) -> list[ProcessingStep]:
        """
        Initializes processing steps with default parameters or overrides.
        Maintains consistent dictionary structure for ProcessingState compatibility.

        Args:
            params_override: Optional dictionary of step-level overrides matching params.yml structure

        Returns:
            List of initialized ProcessingStep instances
        """
        yaml = YAML(typ = 'safe')

        with self.params_file.open('r') as f:
            step_definitions = yaml.load(f)

        # Store steps as instance attribute
        self.steps = []
        params_override = params_override or {}

        for index, step_def in enumerate(step_definitions):
            step_name     = step_def['name']
            step_override = params_override.get(step_name, {})

            # Create parameters list with any overrides
            param_overrides = step_override.get('parameters', {})
            parameters = [
                Parameter(**{
                    **param_def,
                    'value' : param_overrides.get(param_def['name'], param_def['value'])
                })
                for param_def in step_def.get('parameters', [])
            ]

            self.steps.append(
                ProcessingStep(
                    name         = step_name,
                    display_name = step_def['display_name'],
                    toggle_key   = str(index + 1),
                    parameters   = parameters,
                    is_enabled   = step_override.get('enabled', step_def.get('enabled', False))
                )
            )

        return self.steps

    @cache
    def process_image(
        self,
        image_path       : str,
        processing_state : ProcessingState
    ) -> np.ndarray:
        """
        Processes the image according to the enabled processing steps.

        Args:
            image_path        : Path to the image file to process
            processing_state  : ProcessingState instance representing current parameters

        Returns:
            Processed image as numpy array
        """
        image           = self.load_image(image_path)
        processed_image = image.copy()

        for step_name, is_enabled, params in processing_state.steps:
            if not is_enabled:
                continue

            if processing_function :=  PROCESSING_FUNCTIONS.get(step_name):
                params_dict     = {param_name: param_value for param_name, param_value in params}
                processed_image = processing_function(processed_image, params_dict)
            else:
                logger.warning(f"No processing function defined for step '{step_name}'")

        return processed_image

    @cache
    def annotate_image_with_text(
        self,
        image_path       : str,
        processing_state : ProcessingState
    ) -> np.ndarray:
        """
        Annotates the image with recognized text by overlaying bounding boxes and text annotations.
        This visualization helps in verifying the OCR results and understanding where the text was detected.

        Args:
            image_path       : Path to the image to annotate
            processing_state : ProcessingState instance representing current parameters

        Returns:
            Annotated image
        """
        processed_image = self.process_image(image_path, processing_state)
        annotated_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR) if processed_image.ndim == 2 else processed_image.copy()
        ocr_results     = self.extract_text_from_image(image_path, processing_state)

        for bounding_box, text, confidence in ocr_results:
            coordinates = np.array(bounding_box).astype(int)

            # Draw bounding box around detected text
            cv2.polylines(annotated_image, [coordinates], True, (0, 255, 0), 2)
            logger.info(f"OCR Text: '{text}' with confidence {confidence:.2f}")

            # Set text position above the bounding box
            x_coords, y_coords = coordinates[:, 0], coordinates[:, 1]
            text_position = (int(np.mean(x_coords)), int(np.min(y_coords) - 10))

            # Draw the text annotation
            annotated_image = self.draw_text(
                source_image = annotated_image,
                text         = f"{text} ({confidence * 100:.1f}%)",
                position     = text_position
            )

        return annotated_image

    @cache
    def extract_text_from_image(
        self,
        image_path       : str,
        processing_state : ProcessingState
    ) -> list[tuple]:
        """
        Extracts text from a given image using EasyOCR.

        Args:
            image_path       : The path to the image to perform OCR on
            processing_state : ProcessingState instance representing current parameters

        Returns:
            List of tuples containing OCR results
        """
        processed_image = self.process_image(image_path, processing_state)
        try:
            ocr_results = self.reader.readtext(
                processed_image[..., ::-1],  # Convert BGR to RGB
                decoder       = 'greedy',
                rotation_info = [90, 180, 270]
            )
            return ocr_results

        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
            return []

    @staticmethod
    def draw_text(
        source_image : np.ndarray,
        text         : str,
        position     : tuple[int, int],
        scale        : float = 1.0,
        opacity      : float = 0.75
    ) -> np.ndarray:
        """
        Draws bold white text with semi-transparent background.

        Args:
            source_image : Image to draw on
            text         : Text to draw
            position     : (x,y) center position for text box
            scale        : Text size multiplier (1.0 = default size)
            opacity      : Background opacity (0.0 to 1.0)
        """
        # Setup drawing layers
        pil_image  = Image.fromarray(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
        text_layer = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        draw       = ImageDraw.Draw(text_layer)

        # Add thin spaces between characters
        spaced_text = ' '.join(text)

        # Get text dimensions
        font        = ImageFont.load_default()
        bbox        = draw.textbbox((0, 0), spaced_text, font = font)
        text_width  = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Calculate box padding based on scale
        box_padding  = max(4, int(20 * scale * 0.2))
        total_width  = text_width  + box_padding * 2
        total_height = text_height + box_padding * 2

        # Constrain position to keep text box within image bounds
        center_x, center_y = position
        bounded_x = max(total_width  // 2, min(source_image.shape[1] - total_width  // 2, center_x))
        bounded_y = max(total_height // 2, min(source_image.shape[0] - total_height // 2, center_y))

        # Calculate text position relative to center
        text_x = bounded_x - text_width  // 2
        text_y = bounded_y - text_height // 2

        # Draw background
        background_bbox = (
            text_x - box_padding,
            text_y - box_padding,
            text_x + text_width + box_padding,
            text_y + text_height + box_padding
        )
        draw.rectangle(background_bbox, fill = (0, 0, 0, int(255 * opacity)))

        # Draw text multiple times for boldness
        for offset_x, offset_y in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            draw.text(
                (text_x + offset_x, text_y + offset_y),
                spaced_text,
                font = font,
                fill = (255, 255, 255, 255)
            )

        # Composite text layer onto source image
        annotated_image = Image.alpha_composite(pil_image.convert('RGBA'), text_layer)
        return cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGBA2BGR)

    def generate_sidebar_content(
        self,
        steps      : list[ProcessingStep],
        image_name : str
    ) -> list[tuple[str, tuple[int, int, int], float]]:
        """
        Generates a list of sidebar lines with text content, colors, and scaling factors.
        Used internally by render_sidebar to prepare display content.

        Args:
            steps      : List of ProcessingStep instances
            image_name : Name of the current image being processed

        Returns:
            List of tuples containing (text, color, scale_factor) for each line
        """
        lines = [
            (f"[/] Current Image: {image_name}", self.UI_COLORS['TEAL'],  0.85),
            ("", self.UI_COLORS['WHITE'], 1.0)  # Spacer
        ]

        for step in steps:
            status = 'Enabled' if step.is_enabled else 'Disabled'
            lines.append((
                f"[{step.toggle_key}] {step.display_name}: {status}",
                self.UI_COLORS['WHITE'],
                1.0
            ))

            for param in step.parameters:
                lines.append((
                    f"   [{param.decrease_key} | {param.increase_key}] {param.display_name}: {param.display_value}",
                    self.UI_COLORS['GRAY'],
                    0.85
                ))
            lines.append(("", self.UI_COLORS['WHITE'], 1.0))  # Spacer

        lines.append(("[q] Quit", self.UI_COLORS['WHITE'], 1.0))
        return lines

    def render_sidebar(
        self,
        steps         : list[ProcessingStep],
        image_name    : str,
        window_height : int
    ) -> np.ndarray:
        """
        Renders the sidebar image with controls and settings.
        """
        lines             = self.generate_sidebar_content(steps, image_name)
        num_lines         = len(lines)
        margin            = int(0.05 * window_height)
        horizontal_margin = 20
        line_height       = max(20, min(int((window_height - 2 * margin) / num_lines), 40))
        font_scale        = line_height / 40
        font_thickness    = max(1, int(font_scale * 1.5))

        # Determine maximum text width
        max_text_width = max(
            cv2.getTextSize(text, self.FONT_FACE, font_scale * rel_scale, font_thickness)[0][0]
            for text, _, rel_scale in lines if text
        )

        sidebar_width = max_text_width + 2 * horizontal_margin
        sidebar       = np.zeros((window_height, sidebar_width, 3), dtype = np.uint8)
        y_position    = margin + line_height

        # Draw text lines onto the sidebar
        for text, color, rel_scale in lines:
            if text:
                cv2.putText(
                    img       = sidebar,
                    text      = text,
                    org       = (horizontal_margin, y_position),
                    fontFace  = self.FONT_FACE,
                    fontScale = font_scale * rel_scale,
                    color     = color,
                    thickness = font_thickness,
                    lineType  = cv2.LINE_AA
                )
            y_position +=  line_height

        return sidebar

    def extract_text_headless(self, image_files: list[Path]) -> dict:
        """
        Processes a list of images and extracts text from them in headless mode.

        Args:
            image_files : List of image file paths to process.

        Returns:
            Dictionary mapping image names to their OCR results.
        """
        results          = {}
        processing_state = ProcessingState.from_steps(self.steps)

        for image_path in image_files:
            image_name = image_path.name

            try:
                ocr_results = self.extract_text_from_image(
                    str(image_path),
                    processing_state
                )
                results[image_name] = [
                    (text, confidence) for _, text, confidence in ocr_results
                ]

            except Exception as e:
                logger.error(f"Failed to process image {image_name}: {e}")
                continue

        return results

    def run_headless(
        self,
        image_files     : list[Path],
        params_override : dict | None = None
    ):
        """
        Processes images in headless mode.

        Args:
            image_files     : List of image file paths to process
            params_override : Optional parameter overrides
        """
        if not image_files:
            raise ValueError("No image files provided")

        # Initialize or update processing steps
        if not self.steps or params_override:
            self.initialize_steps(params_override)

        results = self.extract_text_headless(image_files)

        if self.output_json:
            with self.output_file.open('w', encoding = 'utf-8') as f:
                json.dump(results, f, ensure_ascii = False, indent = 4)
            logger.info(f"OCR results saved to {self.output_file}")

    def interactive_experiment(
        self,
        image_files     : list[Path],
        params_override : dict | None = None
    ):
        """
        Runs the interactive experiment allowing parameter adjustment and image processing.

        Args:
            image_files     : List of image file paths to process
            params_override : Optional parameter overrides
        """
        if not image_files:
            raise ValueError("No image files provided")

        # Initialize or update processing steps
        if not self.steps or params_override:
            self.initialize_steps(params_override)

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_KEEPRATIO)

        try:
            while True:
                image_path                = image_files[self.state.image_idx]
                self.state.image_name     = image_path.name
                self.state.window_height  = self.DEFAULT_HEIGHT

                processing_state = ProcessingState.from_steps(self.steps)

                # Annotate image with OCR text if OCR is enabled
                ocr_enabled = any(step.is_enabled and step.name ==  'ocr' for step in self.steps)
                if ocr_enabled:
                    display_image = self.annotate_image_with_text(
                        str(image_path),
                        processing_state
                    )
                else:
                    display_image = self.process_image(str(image_path), processing_state)

                # Prepare image for display
                display_image = self.center_image_in_square(display_image)
                if display_image.ndim == 2:
                    display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)

                # Scale image to window height
                display_scale = self.state.window_height / display_image.shape[0]
                display_image = cv2.resize(
                    display_image,
                    (int(display_image.shape[1] * display_scale), self.state.window_height)
                )

                # Add sidebar and display
                sidebar_image  = self.render_sidebar(self.steps, self.state.image_name, self.state.window_height)
                combined_image = np.hstack([display_image, sidebar_image])
                cv2.imshow(self.WINDOW_NAME, combined_image)
                cv2.resizeWindow(self.WINDOW_NAME, combined_image.shape[1], combined_image.shape[0])

                # Handle user input
                key = cv2.waitKey(1) & 0xFF
                if key == 255:
                    continue

                try:
                    char = chr(key)
                except ValueError:
                    continue  # Non-ASCII key pressed

                if char == 'q':
                    break
                elif char == '/':
                    self.state.next_image(len(image_files))
                else:
                    # Handle parameter adjustments
                    for step in self.steps:
                        if char == step.toggle_key:
                            logger.info(step.toggle())
                            break

                        action = step.adjust_param(char)
                        if action:
                            logger.info(action)
                            break

        finally:
            cv2.destroyAllWindows()
