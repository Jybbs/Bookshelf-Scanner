import cv2
import json
import numpy as np

from bookshelf_scanner import ModuleLogger, Utils
from dataclasses       import dataclass
from functools         import cache
from pathlib           import Path
from typing            import Any

logger = ModuleLogger('approver')()

# -------------------- Data Classes --------------------

@dataclass
class DisplayState:
    """
    Tracks the current state of the approval interface.
    """
    image_idx     : int  = 0     # Index of the current image
    image_name    : str  = ''    # Name of the current image
    new_image     : bool = True  # Flag to indicate if the current image is new
    window_height : int  = 800   # Window height in pixels
    show_processed: bool = True  # Toggle between processed and raw images

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

    def toggle_processed_raw(self):
        """
        Toggles between showing processed and raw images.
        """
        self.show_processed = not self.show_processed
        self.new_image      = True

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

# -------------------- MatchApprover Class --------------------

class MatchApprover:
    """
    Provides an interactive UI to review and approve the best match for each image.
    """

    # -------------------- Class Constants --------------------

    PROJECT_ROOT    = Utils.find_root('pyproject.toml')
    ALLOWED_FORMATS = {'.bmp', '.jpg', '.jpeg', '.png', '.tiff'}
    DEFAULT_HEIGHT  = 800
    FONT_FACE       = cv2.FONT_HERSHEY_DUPLEX
    PROCESSED_DIR   = PROJECT_ROOT / 'bookshelf_scanner' / 'images' / 'processed'
    RAW_DIR         = PROJECT_ROOT / 'bookshelf_scanner' / 'images' / 'books'
    MATCHER_FILE    = PROJECT_ROOT / 'bookshelf_scanner' / 'data' / 'results' / 'matcher.json'
    OUTPUT_FILE     = PROJECT_ROOT / 'bookshelf_scanner' / 'data' / 'results' / 'approvals.json'
    UI_COLORS       = {
        'GRAY'  : (200, 200, 200),
        'TEAL'  : (255, 255, 0),
        'WHITE' : (255, 255, 255)
    }
    WINDOW_NAME     = 'Approval Layer'

    # -------------------- Initialization --------------------

    def __init__(
        self,
        threshold    : float       = 0.85,
        window_height: int         = DEFAULT_HEIGHT,
        matcher_file : Path | None = None,
        output_file  : Path | None = None
    ):
        """
        Initializes the MatchApprover instance.

        Args:
            threshold     : Minimum confidence threshold for displaying matches
            window_height : Default window height for UI display
            matcher_file  : Path to the matcher results JSON file
            output_file   : Path to save approvals
        """
        self.threshold     = threshold
        self.window_height = window_height
        self.matcher_file  = matcher_file or self.MATCHER_FILE
        self.output_file   = output_file or self.OUTPUT_FILE
        self.state         = None  # Set in interactive mode
        self.all_matches   = self.load_matches()
        self.image_files   = self.find_image_files(self.PROCESSED_DIR)  # Processed images as baseline
        self.approvals     = {}    # Store user approvals: {image_name: {title, author, score} or None if skipped}

    # -------------------- Data Loading --------------------

    def load_matches(self) -> dict[str, Any]:
        """
        Loads match results from the matcher JSON file.

        Returns:
            dict: Dictionary of match results keyed by image name.
        """
        if not self.matcher_file.is_file():
            logger.error(f"Matcher file not found: {self.matcher_file}")
            return {}
        with self.matcher_file.open('r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    @classmethod
    @cache
    def find_image_files(cls, directory: Path) -> list[Path]:
        """
        Retrieves a sorted list of image files from the given directory.

        Args:
            directory: The directory containing images.

        Returns:
            list: Image file paths

        Raises:
            FileNotFoundError: If no image files are found in the specified directory
        """
        if not directory.is_dir():
            raise FileNotFoundError(f"Image directory not found: {directory}")

        image_files = sorted(
            file for file in directory.glob('*')
            if file.is_file() and file.suffix.lower() in cls.ALLOWED_FORMATS
        )

        if not image_files:
            raise FileNotFoundError(f"No image files found in {directory}")

        return image_files

    # -------------------- Interactive Mode --------------------

    def run_interactive_mode(self):
        """
        Runs the interactive approval UI, allowing users to navigate images and pick matches.
        """
        if not self.image_files:
            raise ValueError("No image files available.")

        self.state = DisplayState(window_height = self.window_height)
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
        logger.info("Starting approval interactive mode.")

        try:
            while True:
                current_image_path    = self.get_current_image_path()
                self.state.image_name = current_image_path.name

                # Load and prepare the displayed image
                display_image  = self.prepare_display_image(current_image_path)
                sidebar_image  = self.render_sidebar(image_name = self.state.image_name)
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

                should_quit = self.process_user_input(char = char)
                if should_quit:
                    break

        finally:
            cv2.destroyAllWindows()

        self.save_approvals()

    def get_current_image_path(self) -> Path:
        """
        Returns the currently selected image path based on state and show_processed flag.

        Returns:
            Path: The current image path
        """
        base_files = self.find_image_files(self.PROCESSED_DIR if self.state.show_processed else self.RAW_DIR)
        return base_files[self.state.image_idx]

    # -------------------- UI Rendering --------------------

    def prepare_display_image(self, image_path: Path) -> np.ndarray:
        """
        Prepares the image for display by resizing and centering it.

        Args:
            image_path: Path to the image

        Returns:
            np.ndarray: Display-ready image
        """
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Unable to load image: {image_path}")
            # Create a blank fallback image
            image = np.zeros((self.window_height, self.window_height, 3), dtype=np.uint8)

        # Center image in a square and scale to window height
        height, width = image.shape[:2]
        max_side      = max(height, width)
        canvas        = np.zeros((max_side, max_side, 3), dtype=np.uint8)
        y_offset      = (max_side - height) // 2
        x_offset      = (max_side - width)  // 2
        canvas[y_offset:y_offset + height, x_offset:x_offset + width] = image

        display_scale = self.window_height / canvas.shape[0]
        display_image = cv2.resize(canvas, (int(canvas.shape[1] * display_scale), self.window_height))
        return display_image

    def generate_sidebar_content(self, image_name: str) -> list[tuple[str, tuple[int,int,int], float]]:
        """
        Generates the sidebar lines, including navigation instructions and matches.

        Args:
            image_name: Name of the current image

        Returns:
            list: (text, color, scale_factor) tuples for rendering
        """
        lines = [
            (f"Current Image: {image_name}", self.UI_COLORS['TEAL'], 1.0),
            ("   [< | >] Navigate Images",   self.UI_COLORS['GRAY'], 0.85),
            ("   [/] Toggle Processed/Raw",  self.UI_COLORS['GRAY'], 0.85),
            ("",                             self.UI_COLORS['WHITE'], 1.0),
        ]

        # Display matches from matcher.json
        image_matches = self.all_matches.get(image_name, {})
        matches_list  = image_matches.get("matches", [])

        # Filter by threshold and sort by score descending
        filtered = [(m['title'], m['author'], m['score']) for m in matches_list if m['score'] >= self.threshold]
        filtered.sort(key = lambda x: x[2], reverse = True)

        lines.append((f"Matches (≥ {self.threshold * 100:.1f}%):", self.UI_COLORS['TEAL'], 1.0))

        if filtered:
            for i, (title, author, score) in enumerate(filtered, start=1):
                score_percent = f"{score * 100:.1f}%"
                lines.append((f" [{i}] {title} by {author} ({score_percent})", self.UI_COLORS['WHITE'], 0.9))
        else:
            lines.append(("   No matches above threshold", self.UI_COLORS['GRAY'], 0.9))

        lines.append(("",               self.UI_COLORS['WHITE'], 1.0))
        lines.append(("[s] Skip image", self.UI_COLORS['WHITE'], 0.9))
        lines.append(("",               self.UI_COLORS['WHITE'], 1.0))
        lines.append(("[q] Quit",       self.UI_COLORS['WHITE'], 1.0))

        return lines

    def render_sidebar(self, image_name: str) -> np.ndarray:
        """
        Renders the sidebar image with the current matches and controls.

        Args:
            image_name : Name of the current image

        Returns:
            np.ndarray: Sidebar image as a numpy array.
        """
        lines             = self.generate_sidebar_content(image_name = image_name)
        num_lines         = len(lines)
        margin            = int(0.05 * self.window_height)
        horizontal_margin = 20
        line_height       = max(20, min(int((self.window_height - 2 * margin) / num_lines), 40))
        font_scale        = line_height / 40
        font_thickness    = max(1, int(font_scale * 1.5))

        max_text_width = max(
            cv2.getTextSize(text, self.FONT_FACE, font_scale * rel_scale, font_thickness)[0][0]
            for text, _, rel_scale in lines if text
        )

        sidebar_width = max_text_width + 2 * horizontal_margin
        sidebar       = np.zeros((self.window_height, sidebar_width, 3), dtype = np.uint8)
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

    def process_user_input(self, char: str) -> bool:
        """
        Processes user input keys and performs corresponding actions.

        Args:
            char : Input character

        Returns:
            bool: True if user chose to quit, False otherwise.
        """
        if char == 'q':
            logger.info("Quitting approval interactive mode.")
            return True

        elif char == '>':
            old_name     = self.state.image_name
            total_images = len(self.find_image_files(self.PROCESSED_DIR if self.state.show_processed else self.RAW_DIR))
            self.state.advance_to_next_image(total_images = total_images)
            logger.info(f"Switched from '{old_name}' to '{self.state.image_name}'")
            return False

        elif char == '<':
            old_name     = self.state.image_name
            total_images = len(self.find_image_files(self.PROCESSED_DIR if self.state.show_processed else self.RAW_DIR))
            self.state.retreat_to_previous_image(total_images = total_images)
            logger.info(f"Switched from '{old_name}' to '{self.state.image_name}'")
            return False

        elif char == '/':
            # Toggle processed/raw
            old_mode = "Processed" if self.state.show_processed else "Raw"
            self.state.toggle_processed_raw()
            new_mode = "Processed" if self.state.show_processed else "Raw"
            logger.info(f"Toggled display from {old_mode} to {new_mode}")
            return False

        # Handle match selection
        if char.isdigit():
            # User picked a match number
            image_name = self.state.image_name
            matches    = self.all_matches.get(image_name, {}).get('matches', [])
            filtered   = [(m['title'], m['author'], m['score']) for m in matches if m['score'] >= self.threshold]
            filtered.sort(key = lambda x: x[2], reverse = True)

            choice = int(char)
            if 1 <= choice <= len(filtered):
                picked = filtered[choice - 1]
                self.approvals[image_name] = {
                    "title" : picked[0],
                    "author": picked[1],
                    "score" : picked[2]
                }
                logger.info(f"User approved '{picked[0]}' by '{picked[1]}' for {image_name}")
                self.move_to_next_image()
            else:
                logger.info(f"Invalid choice '{char}' for {image_name}. Ignoring.")
            return False

        if char.lower() == 's':
            # Skip this image (no approval recorded)
            image_name = self.state.image_name
            self.approvals[image_name] = None
            logger.info(f"User skipped {image_name}")
            self.move_to_next_image()
            return False

        return False

    def move_to_next_image(self):
        """
        Moves to the next image after recording a choice.
        """
        total_images = len(self.find_image_files(self.PROCESSED_DIR if self.state.show_processed else self.RAW_DIR))
        self.state.advance_to_next_image(total_images = total_images)

    # -------------------- Saving Approvals --------------------

    def save_approvals(self):
        """
        Saves the approvals dictionary to a JSON file.
        """
        with self.output_file.open('w', encoding='utf-8') as f:
            json.dump(self.approvals, f, ensure_ascii = False, indent = 4)
        logger.info(f"Approvals saved to {self.output_file}")
