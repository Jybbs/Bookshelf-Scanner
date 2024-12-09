import cv2
import json
import numpy as np

from bookshelf_scanner import ModuleLogger, Utils
from copy              import deepcopy
from dataclasses       import dataclass, field
from easyocr           import Reader
from functools         import cache
from omegaconf         import OmegaConf
from pathlib           import Path
from PIL               import Image, ImageDraw, ImageFont
from typing            import Any

logger = ModuleLogger('extractor')()

# -------------------- Data Classes --------------------

@dataclass
class DisplayState:
    """
    Tracks the current state of image processing and display.
    """
    image_idx     : int = 0      # Index of the current image
    image_name    : str = ''     # Name of the current image
    new_image     : bool = True  # Flag to indicate if the current image is new
    window_height : int = 800    # Window height in pixels

    def advance_to_next_image(self, total_images: int):
        """
        Cycle to the next image and mark it as new.
        """
        self.image_idx = (self.image_idx + 1) % total_images
        self.new_image = True

    def retreat_to_previous_image(self, total_images: int):
        """
        Cycle to the previous image and mark it as new.
        """
        self.image_idx = (self.image_idx - 1) % total_images
        self.new_image = True

    def check_and_reset_new_image_flag(self) -> bool:
        """
        Checks if the current image is new and resets the flag.

        Returns:
            bool: True if the image is new, False otherwise.
        """
        if self.new_image:
            self.new_image = False
            return True
        return False

@dataclass(frozen = True)
class ConfigState:
    """
    Immutable state representing all processing parameters for an image.
    Used as a cache key for consistent image processing results.
    """
    config_dict    : dict = field(compare = False, hash = False)
    config_tuple   : tuple
    param_key_map  : dict[str, tuple[str, str, bool]] = field(compare = False, hash = False)
    step_index_map : dict[int, str]                   = field(compare = False, hash = False)

    def adjust_parameter(self, key_char: str) -> tuple['ConfigState', str | None]:
        """
        Returns a new ConfigState with the specified parameter adjusted, if applicable.

        Args:
            key_char: The character representing the key pressed

        Returns:
            tuple: (new ConfigState, action description or None)
        """
        match = self.param_key_map.get(key_char)
        if not match:
            return self, None

        step_name, param_name, increase = match
        new_dict = deepcopy(self.config_dict)

        param_definition = new_dict["steps"][step_name]["parameters"][param_name]
        old_value        = param_definition["value"]
        delta            = param_definition["step"] if increase else -param_definition["step"]
        new_value        = old_value + delta

        if isinstance(old_value, float):
            new_value = round(
                max(param_definition["min"], min(new_value, param_definition["max"])), 
                2
            )
        else:
            new_value = max(param_definition["min"], min(new_value, param_definition["max"]))

        param_definition["value"] = new_value
        new_state   = ConfigState.from_dict(config_dict = new_dict)
        action_type = 'Increased' if increase else 'Decreased'
        action      = f"{action_type} '{param_definition['display_name']}' from {old_value} to {param_definition['value']}"
        return new_state, action

    @classmethod
    def from_config(cls, config: Any) -> 'ConfigState':
        """
        Convert the OmegaConf config into a stable structure, storing both
        a hashable tuple and the direct dictionary for easy access.

        Args:
            config : The OmegaConf config object.

        Returns:
            ConfigState: Immutable state instance.
        """
        config_dict = OmegaConf.to_container(config, resolve=True)
        return cls.from_dict(config_dict = config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ConfigState':
        """
        Creates a ConfigState instance directly from a configuration dictionary.

        Args:
            config_dict: A dictionary representing the configuration.

        Returns:
            ConfigState: Immutable state instance created from the given dictionary.
        """

        def convert_to_hashable_tuple(value: Any) -> Any:
            """
            Converts a nested dictionary/list structure into a nested tuple structure
            that is deterministic and hashable, ensuring that ConfigState instances
            can serve as stable cache keys.
            """
            if isinstance(value, dict):
                return tuple(sorted((k, convert_to_hashable_tuple(v)) for k, v in value.items()))
            elif isinstance(value, list):
                return tuple(convert_to_hashable_tuple(x) for x in value)
            return value
    
        step_index_map = {i+1: step_name for i, step_name in enumerate(config_dict["steps"].keys())}
        param_key_map  = {}

        for step_name, step_definition in config_dict["steps"].items():
            if "parameters" in step_definition and step_definition["parameters"] is not None:
                for param_name, param_definition in step_definition["parameters"].items():
                    inc_key = param_definition["increase_key"]
                    param_key_map[inc_key]         = (step_name, param_name, True)
                    param_key_map[inc_key.lower()] = (step_name, param_name, False)

        return cls(
            config_dict    = config_dict,
            config_tuple   = convert_to_hashable_tuple(config_dict),
            param_key_map  = param_key_map,
            step_index_map = step_index_map
        )
    
    def extract_parameter_space(self) -> list[dict[str, Any]]:
        """
        Extracts configuration range definitions from this ConfigState.
        
        Returns:
            list: List of dictionaries defining each parameter's bounds, step, and type.
        """
        return [
            {
                'name'       : f"{step_name}.{param_name}",
                'min_value'  : float(param_definition["min"]),
                'max_value'  : float(param_definition["max"]),
                'step_value' : float(param_definition["step"]),
                'is_integer' : isinstance(param_definition["value"], int)
            }
            for step_name, step_definition in self.config_dict["steps"].items()
            if step_definition.get("parameters") is not None
            for param_name, param_definition in step_definition["parameters"].items()
        ]

    def toggle_step_enabled(self, step_index: int) -> tuple['ConfigState', str]:
        """
        Returns a new ConfigState with the enabled state of the specified step toggled.

        Args:
            step_index: Index of the step (1-based)

        Returns:
            tuple: (new ConfigState, action description)
        """
        step_name = self.step_index_map.get(step_index)
        if step_name is None:
            return self, ""

        new_dict    = deepcopy(self.config_dict)
        current_val = new_dict["steps"][step_name]["enabled"]
        new_dict["steps"][step_name]["enabled"] = not current_val

        new_state       = ConfigState.from_dict(config_dict = new_dict)
        step_definition = new_state.config_dict["steps"][step_name]
        action          = f"'{step_definition['display_name']}' {'Enabled' if step_definition['enabled'] else 'Disabled'}"
        return new_state, action

    def __hash__(self):
        return hash(self.config_tuple)

# -------------------- Processing Functions --------------------

def adjust_brightness(input_image: np.ndarray, parameters: dict) -> np.ndarray:
    """
    Adjusts the brightness of the image, enhancing text visibility.
    """
    value     = parameters['brightness_value']['value']
    hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 2] = cv2.add(hsv_image[:, :, 2], value)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def adjust_contrast(input_image: np.ndarray, parameters: dict) -> np.ndarray:
    """
    Adjusts the contrast of the image to enhance text-background distinction.
    """
    alpha = parameters['contrast_value']['value']
    return cv2.convertScaleAbs(input_image, alpha = alpha, beta = 0)

def apply_clahe(input_image: np.ndarray, parameters: dict) -> np.ndarray:
    """
    Applies CLAHE to enhance local contrast and reveal text details.
    """
    clip_limit = parameters['clahe_clip_limit']['value']
    lab_image  = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)
    clahe      = cv2.createCLAHE(clipLimit = clip_limit)

    lab_image[:, :, 0] = clahe.apply(lab_image[:, :, 0])
    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

def remove_shadow(input_image: np.ndarray, parameters: dict) -> np.ndarray:
    """
    Removes shadows from the image to improve text isolation.
    """
    kernel_size = int(parameters['shadow_kernel_size']['value']) | 1
    median_blur = int(parameters['shadow_median_blur']['value']) | 1
    kernel      = np.ones((kernel_size, kernel_size), np.uint8)
    channels    = list(cv2.split(input_image))

    for i in range(len(channels)):
        dilated     = cv2.dilate(channels[i], kernel)
        bg_image    = cv2.medianBlur(dilated, median_blur)
        diff_image  = 255 - cv2.absdiff(channels[i], bg_image)
        channels[i] = cv2.normalize(diff_image, None, 0, 255, cv2.NORM_MINMAX)

    return cv2.merge(channels)

def rotate_image(input_image: np.ndarray, parameters: dict) -> np.ndarray:
    """
    Rotates the image by a specified angle to correct text orientation.
    """
    angle = parameters['rotation_angle']['value']
    if angle % 360 != 0:
        rotations = int(angle / 90) % 4
        return np.rot90(input_image, rotations)
    return input_image

PROCESSING_FUNCTIONS = {
    'image_rotation'        : rotate_image,
    'shadow_removal'        : remove_shadow,
    'color_clahe'           : apply_clahe,
    'brightness_adjustment' : adjust_brightness,
    'contrast_adjustment'   : adjust_contrast
}

# -------------------- TextExtractor Class --------------------


class TextExtractor:
    """
    Handles image processing and text extraction using OCR.
    """

    # -------------------- Class Constants --------------------

    PROJECT_ROOT    = Utils.find_root('pyproject.toml')
    ALLOWED_FORMATS = {'.bmp', '.jpg', '.jpeg', '.png', '.tiff'}
    DEFAULT_HEIGHT  = 800
    FONT_FACE       = cv2.FONT_HERSHEY_DUPLEX
    IMAGE_DIR       = PROJECT_ROOT / 'bookshelf_scanner' / 'images' / 'processed'
    OUTPUT_FILE     = PROJECT_ROOT / 'bookshelf_scanner' / 'data' / 'results' / 'extractor.json'
    PARAMS_FILE     = PROJECT_ROOT / 'bookshelf_scanner' / 'config' / 'extractor.yml'
    UI_COLORS       = {
        'GRAY'  : (200, 200, 200),
        'TEAL'  : (255, 255, 0),
        'WHITE' : (255, 255, 255)
    }
    WINDOW_NAME     = 'Bookshelf Scanner'

    # -------------------- Initialization and Configuration Methods --------------------

    def __init__(
        self,
        allowed_formats : set[str] | None = None,
        config_file     : Path | None     = None,
        output_file     : Path | None     = None,
        output_json     : bool            = False,
        output_images   : bool            = False,
        image_dir       : Path | None     = None,
        window_height   : int             = DEFAULT_HEIGHT
    ):
        """
        Initializes the TextExtractor instance.

        Args:
            allowed_formats : Set of allowed image extensions (defaults to ALLOWED_FORMATS)
            config_file     : Optional custom path to config.yml
            output_file     : Path to save resultant strings from OCR processing
            output_json     : Whether to output OCR results to JSON file
            output_images   : Whether to output annotated images
            image_dir       : Directory to save annotated images if output_images is True
            window_height   : Default window height for UI display (relevant for interactive mode)
        """
        self.allowed_formats   = allowed_formats or self.ALLOWED_FORMATS
        self.collected_results = {}
        self.config_file       = config_file or self.PARAMS_FILE
        self.output_file       = output_file or self.OUTPUT_FILE
        self.output_json       = output_json
        self.output_images     = output_images
        self.image_dir         = image_dir
        self.window_height     = window_height
        self.state             = None  # Will be set in interactive mode

        if self.output_images and self.image_dir is not None:
            self.image_dir.mkdir(parents=True, exist_ok=True)

        # Load base config and setup the Reader
        self.base_config  = OmegaConf.load(self.config_file)
        self.config_space = ConfigState.from_config(self.base_config).extract_parameter_space()
        self.reader       = Reader(
            lang_list = self.base_config["easyocr"]["language_list"], 
            gpu       = self.base_config["easyocr"]["gpu_enabled"]
        )

    def merge_steps_config(self, config_override: dict | None = None) -> ConfigState:
        """
        Merges the base configuration with any provided overrides and returns a new ConfigState.

        Args:
            config_override: Optional dictionary of step-level overrides.

        Returns:
            ConfigState: The newly created configuration state after merging.
        """
        merged_config = OmegaConf.merge(self.base_config, OmegaConf.create(config_override)) if config_override else self.base_config
        return ConfigState.from_config(config = merged_config)

    # -------------------- Headless Mode Operations --------------------

    def run_headless_mode(
        self, 
        image_files     : list[Path], 
        config_override : dict | None = None
    ):
        """
        Processes images in a non-interactive (headless) mode.

        Args:
            config_override : Optional parameter overrides
            image_files     : List of image file paths to process
        """
        if not image_files:
            raise ValueError("No image files provided")

        config_state = self.merge_steps_config(config_override = config_override)
        results      = {}

        for image_path in image_files:
            image_name = image_path.name
            try:
                ocr_results = self.perform_ocr(config_state = config_state, image_path = str(image_path))
                results[image_name] = {
                    "ocr_results": [
                        {"text": text, "confidence": confidence}
                        for _, text, confidence in ocr_results
                    ]
                }

                # Save annotated image if requested
                if self.output_images and self.image_dir is not None:
                    processed_image = self.process_image(config_state = config_state, image_path = str(image_path))
                    annotated_image = self.annotate_original_image_with_ocr(processed_image, ocr_results)
                    self.save_annotated_image(annotated_image, image_name)

            except Exception as e:
                logger.error(f"Failed to process image {image_name}: {e}")
                continue

        if self.output_json:
            self.save_to_json(results)

        return results

    # -------------------- Image Loading and Preparation --------------------

    @staticmethod
    def center_image_in_square(input_image: np.ndarray) -> np.ndarray:
        """
        Centers the image in a square canvas with sides equal to the longest side.

        Args:
            input_image : Input image.

        Returns:
            Centered square image.
        """
        height, width = input_image.shape[:2]
        max_side      = max(height, width)
        canvas        = np.zeros((max_side, max_side, 3), dtype=np.uint8)
        y_offset      = (max_side - height) // 2
        x_offset      = (max_side - width) // 2

        canvas[y_offset:y_offset + height, x_offset:x_offset + width] = input_image
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

    def prepare_display_image(self, processed_image: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Centers the image, scales it to the window height, and converts it to BGR if needed.

        Args:
            processed_image: The image to prepare for display.

        Returns:
            tuple: (display_image, display_scale)
        """
        centered_image = self.center_image_in_square(input_image = processed_image)
        if centered_image.ndim == 2:
            centered_image = cv2.cvtColor(centered_image, cv2.COLOR_GRAY2BGR)

        display_scale = self.window_height / centered_image.shape[0]
        display_image = cv2.resize(
            centered_image,
            (int(centered_image.shape[1] * display_scale), self.window_height)
        )
        return display_image, display_scale

    # -------------------- Interactive Mode Operations --------------------

    def run_interactive_mode(
        self,
        image_files     : list[Path],
        config_override : dict | None = None
    ):
        """
        Runs the interactive experiment allowing parameter adjustment and image processing.

        Args:
            config_override : Optional parameter overrides
            image_files     : List of image file paths to process
        """
        if not image_files:
            raise ValueError("No image files provided")

        # Set up the display state since we're in interactive mode
        self.state   = DisplayState(window_height = self.window_height)
        config_state = self.merge_steps_config(config_override = config_override)

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
        logger.info("Starting interactive experiment.")

        try:
            while True:
                current_image_path    = image_files[self.state.image_idx]
                self.state.image_name = current_image_path.name
                ocr_results           = self.perform_ocr(config_state = config_state, image_path = str(current_image_path))

                # Collect results for potential JSON saving later
                self.collected_results[self.state.image_name] = {
                    "ocr_results": [
                        {"text": text, "confidence": confidence}
                        for _, text, confidence in ocr_results
                    ]
                }

                # Save annotated image if requested
                if self.output_images and self.image_dir is not None:
                    processed_image = self.process_image(config_state = config_state, image_path = str(current_image_path))
                    annotated_image = self.annotate_original_image_with_ocr(processed_image, ocr_results)
                    self.save_annotated_image(annotated_image, self.state.image_name)

                if self.state.check_and_reset_new_image_flag():
                    self.log_ocr_results(ocr_results = ocr_results)

                processed_image      = self.process_image(config_state = config_state, image_path = str(current_image_path))
                display_image, scale = self.prepare_display_image(processed_image = processed_image)

                if ocr_results:
                    adjusted_ocr_results = self.adjust_ocr_coordinates(
                        display_scale = scale,
                        ocr_results   = ocr_results,
                        original_size = processed_image.shape[:2]
                    )
                    display_image = self.annotate_image_with_ocr(
                        adjusted_ocr_results = adjusted_ocr_results,
                        display_image        = display_image
                    )

                sidebar_image  = self.render_sidebar(config_state = config_state, image_name = self.state.image_name, window_height = self.state.window_height)
                combined_image = np.hstack([display_image, sidebar_image])
                cv2.imshow(self.WINDOW_NAME, combined_image)
                cv2.resizeWindow(self.WINDOW_NAME, combined_image.shape[1], combined_image.shape[0])

                key = cv2.waitKey(1) & 0xFF
                if key == 255:
                    continue

                try:
                    char = chr(key)
                except ValueError:
                    continue

                should_quit, config_state = self.process_user_input(char = char, config_state = config_state, ocr_results = ocr_results, total_images = len(image_files))
                if should_quit:
                    break

        finally:
            cv2.destroyAllWindows()

        if self.output_json and self.collected_results:
            self.save_to_json(self.collected_results)

    # -------------------- OCR Operations --------------------

    @cache
    def perform_ocr(
        self, 
        config_state : ConfigState, 
        image_path   : str
    ) -> list[tuple]:
        """
        Extracts text from a given image using EasyOCR.

        Args:
            config_state : ConfigState instance representing current parameters
            image_path   : The path to the image to perform OCR on

        Returns:
            List of tuples containing OCR results
        """
        processed_image = self.process_image(config_state = config_state, image_path = image_path)
        try:
            decoder       = config_state.config_dict["easyocr"]["decoder"]
            rotation_info = config_state.config_dict["easyocr"]["rotation_info"]

            ocr_results = self.reader.readtext(
                processed_image[..., ::-1],
                decoder       = decoder,
                rotation_info = rotation_info
            )
            logger.debug(f"Extracted text from image '{image_path}'.")
            return ocr_results

        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
            return []

    # -------------------- UI Rendering and Annotation --------------------

    def annotate_image_with_ocr(
        self,
        adjusted_ocr_results : list[tuple],
        display_image        : np.ndarray
    ) -> np.ndarray:
        """
        Annotates the display image with OCR results.

        Args:
            adjusted_ocr_results : List of adjusted OCR results.
            display_image        : The image to annotate.

        Returns:
            np.ndarray: Annotated image.
        """
        annotated_image = display_image.copy()
        for coordinates, text, confidence in adjusted_ocr_results:
            coordinates = np.array(coordinates).astype(int)
            cv2.polylines(annotated_image, [coordinates], True, (0, 255, 0), 2)

            x_coords, y_coords = coordinates[:, 0], coordinates[:, 1]
            text_position      = (int(np.mean(x_coords)), max(int(np.min(y_coords) - 10), 0))

            annotated_image = self.draw_text(
                opacity      = 0.75,
                position     = text_position,
                scale        = 1.0,
                source_image = annotated_image,
                text         = f"{text} ({confidence * 100:.1f}%)"
            )
        return annotated_image

    def annotate_original_image_with_ocr(
        self,
        original_image : np.ndarray,
        ocr_results    : list[tuple[list, str, float]]
    ) -> np.ndarray:
        """
        Annotates the original processed image with OCR results (no sidebar, no scaling).

        Args:
            original_image : The original processed image at full resolution.
            ocr_results    : OCR results with bounding boxes and text.

        Returns:
            np.ndarray: Annotated image.
        """
        annotated_image = original_image.copy()
        for coordinates, text, confidence in ocr_results:
            coordinates = np.array(coordinates).astype(int)
            cv2.polylines(annotated_image, [coordinates], True, (0, 255, 0), 2)

            x_coords, y_coords = coordinates[:, 0], coordinates[:, 1]
            text_position      = (int(np.mean(x_coords)), max(int(np.min(y_coords) - 10), 0))

            annotated_image = self.draw_text(
                opacity      = 0.75,
                position     = text_position,
                scale        = 1.0,
                source_image = annotated_image,
                text         = f"{text} ({confidence * 100:.1f}%)"
            )
        return annotated_image

    @staticmethod
    def draw_text(
        position     : tuple[int, int],
        source_image : np.ndarray,
        text         : str,
        opacity      : float = 0.75,
        scale        : float = 1.0
    ) -> np.ndarray:
        """
        Draws bold white text with semi-transparent background.

        Args:
            opacity      : Background opacity (0.0 to 1.0)
            position     : (x,y) center position for text box
            scale        : Text size multiplier (1.0 = default size)
            source_image : Image to draw on
            text         : Text to draw

        Returns:
            np.ndarray: Image with text drawn on it.
        """
        pil_image  = Image.fromarray(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
        text_layer = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        draw       = ImageDraw.Draw(text_layer)

        spaced_text = ' '.join(text)
        font        = ImageFont.load_default()
        bbox        = draw.textbbox((0, 0), spaced_text, font = font)
        text_width  = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        box_padding  = max(4, int(20 * scale * 0.2))
        total_width  = text_width  + box_padding * 2
        total_height = text_height + box_padding * 2

        center_x, center_y = position
        bounded_x = max(total_width  // 2, min(source_image.shape[1] - total_width  // 2, center_x))
        bounded_y = max(total_height // 2, min(source_image.shape[0] - total_height // 2, center_y))

        text_x = bounded_x - text_width  // 2
        text_y = bounded_y - text_height // 2

        background_bbox = (
            text_x - box_padding,
            text_y - box_padding,
            text_x + text_width + box_padding,
            text_y + text_height + box_padding
        )
        draw.rectangle(background_bbox, fill = (0, 0, 0, int(255 * opacity)))

        for offset_x, offset_y in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            draw.text(
                (text_x + offset_x, text_y + offset_y),
                spaced_text,
                font = font,
                fill = (255, 255, 255, 255)
            )

        annotated_image = Image.alpha_composite(pil_image.convert('RGBA'), text_layer)
        return cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGBA2BGR)

    def generate_sidebar_content(
        self, 
        config_state : ConfigState,
        image_name   : str
    ) -> list[tuple[str, tuple[int, int, int], float]]:
        """
        Generates a list of sidebar lines with text content, colors, and scaling factors.
        Used internally by render_sidebar to prepare display content.

        Args:
            config_state : Current ConfigState
            image_name   : Name of the current image being processed

        Returns:
            list: (text, color, scale_factor) tuples
        """
        lines = [
            (f"Current Image: {image_name}",       self.UI_COLORS['TEAL'], 1.0),
            ("   [< | >] Navigate Between Images", self.UI_COLORS['GRAY'], 0.85),
            ("", self.UI_COLORS['WHITE'], 1.0)
        ]

        steps_dict = config_state.config_dict["steps"]
        for i in sorted(config_state.step_index_map.keys()):
            step_name       = config_state.step_index_map[i]
            step_definition = steps_dict[step_name]
            status          = 'Enabled' if step_definition['enabled'] else 'Disabled'

            # Add step line
            lines.append((
                f"[{i}] {step_definition['display_name']}: {status}",
                self.UI_COLORS['WHITE'],
                1.0
            ))

            # Add parameters if any
            if 'parameters' in step_definition and step_definition['parameters'] is not None:
                for _, param_definition in step_definition['parameters'].items():
                    inc_key   = param_definition['increase_key']
                    dec_key   = inc_key.lower()
                    value_str = (
                        f"{param_definition['value']:.2f}"
                        if isinstance(param_definition['value'], float)
                        else str(param_definition['value'])
                    )
                    lines.append((
                        f"   [{dec_key} | {inc_key}] {param_definition['display_name']}: {value_str}",
                        self.UI_COLORS['GRAY'],
                        0.85
                    ))
            lines.append(("", self.UI_COLORS['WHITE'], 1.0))

        lines.append(("[q] Quit", self.UI_COLORS['WHITE'], 1.0))
        return lines

    def render_sidebar(
        self, 
        config_state  : ConfigState,
        image_name    : str, 
        window_height : int
    ) -> np.ndarray:
        """
        Renders the sidebar image with controls and settings.

        Args:
            config_state  : Current ConfigState
            image_name    : Name of the current image being processed
            window_height : Height of the display window

        Returns:
            np.ndarray: Sidebar image as a numpy array.
        """
        lines             = self.generate_sidebar_content(config_state = config_state, image_name = image_name)
        num_lines         = len(lines)
        margin            = int(0.05 * window_height)
        horizontal_margin = 20
        line_height       = max(20, min(int((window_height - 2 * margin) / num_lines), 40))
        font_scale        = line_height / 40
        font_thickness    = max(1, int(font_scale * 1.5))

        max_text_width = max(
            cv2.getTextSize(text, self.FONT_FACE, font_scale * rel_scale, font_thickness)[0][0]
            for text, _, rel_scale in lines if text
        )

        sidebar_width = max_text_width + 2 * horizontal_margin
        sidebar       = np.zeros((window_height, sidebar_width, 3), dtype=np.uint8)
        y_position    = margin + line_height

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
            y_position += line_height

        return sidebar

    # -------------------- User Input Handling --------------------

    def process_user_input(
        self,
        char         : str,
        config_state : ConfigState,
        ocr_results  : list,
        total_images : int
    ) -> tuple[bool, ConfigState]:
        """
        Processes user input and performs corresponding actions.

        Args:
            char         : Input character
            config_state : Current ConfigState
            ocr_results  : Current OCR results for logging after changes
            total_images : Total number of images for navigation

        Returns:
            tuple: (should_quit, updated_config_state)
        """
        if char == 'q':
            logger.info("Quitting interactive experiment.")
            return True, config_state

        elif char == '>':
            old_name = self.state.image_name
            self.state.advance_to_next_image(total_images = total_images)
            logger.info(f"Switched from '{old_name}' to '{self.state.image_name}'")
            return False, config_state

        elif char == '<':
            old_name = self.state.image_name
            self.state.retreat_to_previous_image(total_images = total_images)
            logger.info(f"Switched from '{old_name}' to '{self.state.image_name}'")
            return False, config_state

        if char.isdigit():
            step_index        = int(char)
            new_state, action = config_state.toggle_step_enabled(step_index = step_index)
            if action:
                config_state = new_state
                logger.info(action)
                self.log_ocr_results(ocr_results = ocr_results)
            return False, config_state

        new_state, action = config_state.adjust_parameter(key_char = char)
        if action:
            config_state = new_state
            logger.info(action)
            self.log_ocr_results(ocr_results = ocr_results)
        return False, config_state

    # -------------------- Utility Methods --------------------

    def adjust_ocr_coordinates(
        self,
        display_scale : float,
        ocr_results   : list[tuple],
        original_size : tuple[int, int]
    ) -> list[tuple]:
        """
        Adjust OCR coordinates to match the display image.

        Args:
            display_scale : Scale factor used for the display image.
            ocr_results   : List of OCR results (bounding_box, text, confidence).
            original_size : Original image size (height, width).

        Returns:
            list: Adjusted OCR results.
        """
        orig_height, orig_width = original_size
        max_side                = max(orig_height, orig_width)
        y_offset                = (max_side - orig_height) // 2
        x_offset                = (max_side - orig_width)  // 2

        adjusted_ocr_results = []
        for bounding_box, text, confidence in ocr_results:
            adjusted_box = [
                [
                    (x + x_offset) * display_scale,
                    (y + y_offset) * display_scale
                ]
                for x, y in bounding_box
            ]
            adjusted_ocr_results.append((adjusted_box, text, confidence))

        return adjusted_ocr_results

    @classmethod
    def find_image_files(cls, subdirectory: str = 'books') -> list[Path]:
        """
        Retrieves a sorted list of image files from a subdirectory within the images folder.

        Args:
            subdirectory : Name of the subdirectory within 'images' to process (e.g., 'books', 'Bookcases', 'Shelves')

        Returns:
            list: Image file paths

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

    @cache
    def process_image(
        self,
        config_state : ConfigState,
        image_path   : str
    ) -> np.ndarray:
        """
        Processes the image according to the enabled processing steps.

        Args:
            config_state : ConfigState instance representing current parameters
            image_path   : Path to the image file to process

        Returns:
            np.ndarray: Processed image
        """
        image           = self.load_image(image_path = image_path)
        processed_image = image.copy()
        steps_dict      = config_state.config_dict['steps']

        for step_name, step_definition in steps_dict.items():
            if not step_definition.get('enabled', False):
                continue

            if 'parameters' in step_definition:
                parameters = {
                    name: {'value': p['value']}
                    for name, p in step_definition['parameters'].items()
                }
            else:
                parameters = {}

            processing_function = PROCESSING_FUNCTIONS.get(step_name)
            if processing_function:
                processed_image = processing_function(processed_image, parameters)
                logger.debug(f"Applied '{step_name}' step.")
            else:
                logger.warning(f"No processing function defined for step '{step_name}'")

        return processed_image

    def log_ocr_results(self, ocr_results: list[tuple[list, str, float]]):
        """
        Logs OCR results in a consistent format. Logs only when in interactive mode (self.state is not None).

        Args:
            ocr_results: List of OCR results (bounding_box, text, confidence)
        """
        if ocr_results and self.state is not None:
            logger.info(f"OCR Results for image '{self.state.image_name}':")
            for _, text, confidence in ocr_results:
                logger.info(f"Text: '{text}' with confidence {confidence:.2f}")

    def save_to_json(self, results: dict):
        """
        Saves OCR results to a JSON file if output_json is True.

        Args:
            results : Dictionary of OCR results keyed by image name.
        """
        if not self.output_json:
            return

        with self.output_file.open('w', encoding = 'utf-8') as f:
            json.dump(results, f, ensure_ascii = False, indent = 4)
        logger.info(f"OCR results saved to {self.output_file}")

    def save_annotated_image(self, annotated_image: np.ndarray, image_name: str):
        """
        Saves the annotated image to the specified directory.

        Args:
            annotated_image : The annotated image to save
            image_name      : The file name for the output image
        """
        if self.image_dir is not None:
            output_path = self.image_dir / image_name
            cv2.imwrite(str(output_path), annotated_image)
            logger.info(f"Annotated image saved to {output_path}")
