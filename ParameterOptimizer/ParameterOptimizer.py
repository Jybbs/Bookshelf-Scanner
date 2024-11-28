import json
import logging
import itertools
import numpy as np

from dataclasses   import dataclass, field
from pathlib       import Path
from ruamel.yaml   import YAML
from typing        import Any, Iterator, Optional

from TextExtractor import TextExtractor

# -------------------- Configuration and Logging --------------------

logger = logging.getLogger('ParameterOptimizer')
logger.setLevel(logging.INFO)

handler = logging.FileHandler(Path(__file__).parent / 'ParameterOptimizer.log', mode = 'w')
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger.addHandler(handler)
logger.propagate = False

# -------------------- Data Classes --------------------

@dataclass
class OptimizationConfig:
    """
    Configuration for the optimization process, including parameter ranges
    and settings for batch processing.
    """
    enabled_steps  : list[str]                              # List of processing step names to optimize
    param_ranges   : dict[str, tuple[float, float, float]]  # Parameter ranges for optimization
    save_frequency : int = 10                               # How often to save intermediate results
    batch_size     : int = 100                              # Number of combinations to process per batch

    def calculate_total_combinations(self) -> int:
        """
        Calculates the total number of parameter combinations.

        Returns:
            Total number of possible combinations
        """
        total = 1
        for min_val, max_val, step in self.param_ranges.values():
            num_values = int((max_val - min_val) / step) + 1
            total *= num_values
        return total

@dataclass
class ImageOptimizationResult:
    """
    Stores the best optimization results for a single image, including
    the parameters used and the resulting OCR text and metrics.
    """
    parameters   : dict[str, Any]          # Parameters that achieved best results
    text_results : list[tuple[str, float]] # OCR results for the image
    score        : float                   # Total character-weighted confidence score
    char_count   : int                     # Total number of characters recognized
    iterations   : int = 0                 # Number of optimization iterations run

    def update_if_better(
        self,
        new_params    : dict[str, Any],
        new_results   : list[tuple[str, float]],
        iteration_num : int
    ) -> bool:
        """
        Updates result if new parameters achieve better score.

        Args:
            new_params    : New parameter set to evaluate
            new_results   : New OCR results to evaluate
            iteration_num : Current optimization iteration

        Returns:
            True if results were updated, False otherwise
        """
        new_score, new_count = ParameterOptimizer.calculate_score(new_results)

        if new_score > self.score:
            self.parameters   = new_params.copy()
            self.text_results = new_results
            self.score        = new_score
            self.char_count   = new_count
            self.iterations   = iteration_num
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """
        Converts optimization result to dictionary format.

        Returns:
            Dictionary containing optimization results and metrics
        """
        return {
            "best_parameters": self.parameters,
            "text_results"   : self.text_results,
            "metrics"        : {
                "score"      : round(self.score, 4),
                "char_count" : self.char_count,
                "iterations" : self.iterations
            }
        }

@dataclass
class OptimizerState:
    """
    Maintains the state of the optimization process, including the best
    results per image and tracking of current iteration and batch.
    """
    best_results  : dict[str, ImageOptimizationResult] = field(default_factory = dict)
    iteration     : int = 0   # Current iteration number

# -------------------- ParameterOptimizer Class --------------------

class ParameterOptimizer:
    """
    Optimizes the parameters for text extraction by testing various
    combinations and recording the best results.
    """
    BATCH_SIZE     : int  = 100
    OUTPUT_FILE    : Path = Path(__file__).parent / 'optimized_results.json'
    PARAMS_FILE    : Path = Path(__file__).resolve().parent.parent / 'config' / 'params.yml'
    SAVE_FREQUENCY : int  = 10

    def __init__(
        self,
        extractor      : TextExtractor,
        batch_size     : int            = BATCH_SIZE,
        output_file    : Path | None    = None,
        params_file    : Path | None    = None,
        save_frequency : int            = SAVE_FREQUENCY
    ):
        """
        Initializes the ParameterOptimizer instance.

        Args:
            extractor      : An instance of TextExtractor to perform OCR
            batch_size     : Number of parameter combinations to process per batch
            output_file    : Path to save optimization results
            params_file    : Path to parameter configuration file
            save_frequency : How often to save intermediate results
        """
        self.extractor      = extractor
        self.batch_size     = batch_size
        self.output_file    = output_file or self.OUTPUT_FILE
        self.params_file    = params_file or self.PARAMS_FILE
        self.save_frequency = save_frequency

        self.state              = OptimizerState()
        self.config             = self.load_config()
        self.total_combinations = self.config.calculate_total_combinations()
        logger.info(f"Initialized with {self.total_combinations:,} parameter combinations to test")

    def load_config(self) -> OptimizationConfig:
        """
        Loads and validates optimization configuration from params file.

        Returns:
            Validated OptimizationConfig instance

        Raises:
            FileNotFoundError: If the configuration file cannot be found
        """
        yaml = YAML(typ = 'safe')

        try:
            with self.params_file.open('r') as f:
                step_definitions = yaml.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.params_file}")
            raise

        logger.info(f"Loading configuration from {self.params_file}")

        # Get all available steps and parameter ranges
        enabled_steps = []
        param_ranges  = {}

        for step_def in step_definitions:
            step_name = step_def['name']
            if step_name != 'ocr':  # Skip OCR as it's always enabled
                enabled_steps.append(step_name)

                for param in step_def.get('parameters', []):
                    if all(k in param for k in ('name', 'min', 'max', 'step')):
                        param_ranges[param['name']] = (
                            float(param['min']),
                            float(param['max']),
                            float(param['step'])
                        )

        return OptimizationConfig(
            enabled_steps  = enabled_steps,
            param_ranges   = param_ranges,
            save_frequency = self.save_frequency,
            batch_size     = self.batch_size
        )

    def generate_parameter_combinations(self) -> Iterator[dict[str, Any]]:
        """
        Generates valid parameter combinations based on enabled steps.
        Ensures OCR is always on and only generates parameter combinations
        for enabled processing steps.

        Yields:
            Dictionary containing a valid parameter combination
        """
        non_ocr_steps          = self.config.enabled_steps
        param_ranges           = self.config.param_ranges
        combinations_generated = 0

        # Generate all possible combinations of enabled/disabled steps
        for r in range(1, len(non_ocr_steps) + 1):
            for step_combo in itertools.combinations(non_ocr_steps, r):
                # Create base combination with OCR always enabled
                base_combo = {'use_ocr': True}

                # Set which steps are enabled in this combination
                for step_name in non_ocr_steps:
                    base_combo[f"use_{step_name}"] = step_name in step_combo

                # Get parameters for enabled steps
                enabled_params = {}
                for step_name in step_combo:
                    # Find parameters belonging to this step
                    for param_name, param_range in param_ranges.items():
                        if param_name.startswith(step_name):
                            min_val, max_val, step_size = param_range
                            enabled_params[param_name] = np.arange(
                                start = min_val,
                                stop  = max_val + step_size / 2,
                                step  = step_size
                            ).tolist()

                if not enabled_params:
                    combinations_generated += 1
                    yield base_combo
                    continue

                param_names        = list(enabled_params.keys())
                value_combinations = itertools.product(*enabled_params.values())

                for values in value_combinations:
                    combinations_generated += 1
                    combo = base_combo.copy()
                    combo.update(dict(zip(param_names, values)))
                    yield combo

    @staticmethod
    def calculate_score(text_results: list[tuple[str, float]]) -> tuple[float, int]:
        """
        Calculates confidence-weighted character score for OCR results.

        Args:
            text_results: List of (text, confidence) tuples from OCR

        Returns:
            Tuple of (total_score, total_character_count)
        """
        if not text_results:
            return 0.0, 0

        total_score = 0.0
        char_count  = 0

        for text, confidence in text_results:
            text_chars   = len(text.strip())
            char_weight  = 1.0  # Adjust if needed
            total_score += text_chars * confidence * char_weight
            char_count  += text_chars

        return total_score, char_count

    def save_results(self):
        """
        Saves current optimization results to the output file.
        """
        output_dict = {
            name: result.to_dict()
            for name, result in sorted(self.state.best_results.items())
        }

        with self.output_file.open('w', encoding = 'utf-8') as f:
            json.dump(output_dict, f, ensure_ascii = False, indent = 4)

        if self.state.iteration >= self.total_combinations:
            logger.info(f"Results saved to {self.output_file}")
            scores = [result.score for result in self.state.best_results.values()]
            chars  = [result.char_count for result in self.state.best_results.values()]

            if scores:
                logger.info(f"Final Results Summary:")
                logger.info(f"  Images processed : {len(scores)}")
                logger.info(f"  Average score    : {sum(scores)/len(scores):.2f}")
                logger.info(f"  Average chars    : {sum(chars)/len(chars):.1f}")
                logger.info(f"  Best score       : {max(scores):.2f}")
                logger.info(f"  Worst score      : {min(scores):.2f}")

    def process_combination(
        self,
        params      : dict[str, Any],
        image_files : list[Path]
    ):
        """
        Processes a single parameter combination across all images.

        Args:
            params      : Dictionary of parameter settings to test
            image_files : List of image file paths to process
        """
        self.state.iteration += 1

        # Set parameters in extractor
        self.extractor.initialize_steps(params_override = params)

        # Extract text using extractor's headless method
        results = self.extractor.extract_text_headless(image_files)

        # Track improvements
        improvements = 0

        # Update best results for each image
        for image_name, ocr_results in results.items():
            if image_name not in self.state.best_results:
                score, count = self.calculate_score(ocr_results)
                self.state.best_results[image_name] = ImageOptimizationResult(
                    parameters   = params.copy(),
                    text_results = ocr_results,
                    score        = score,
                    char_count   = count,
                    iterations   = self.state.iteration
                )
                improvements += 1
            else:
                if self.state.best_results[image_name].update_if_better(
                    new_params    = params,
                    new_results   = ocr_results,
                    iteration_num = self.state.iteration
                ):
                    logger.info(
                        f"Image {image_name} | Score improved to {self.state.best_results[image_name].score:.2f} "
                        f"({self.state.best_results[image_name].char_count} chars)"
                    )
                    improvements += 1

        # Only log progress periodically or when improvements occur
        if improvements > 0 or self.state.iteration % 100 == 0:
            progress = (self.state.iteration / self.total_combinations) * 100
            logger.info(
                f"Progress: {progress:.1f}% - "
                f"Combination {self.state.iteration:,}/{self.total_combinations:,} "
                f"improved {improvements} images"
            )

    def optimize(
        self,
        image_files : list[Path],
        resume      : bool = False
    ) -> dict[str, ImageOptimizationResult]:
        """
        Runs the optimization process on the provided image files.

        Args:
            image_files : List of image file paths to process
            resume      : Whether to resume from previous results

        Returns:
            Dictionary of best results per image
        """
        if not image_files:
            raise ValueError("No image files provided")

        logger.info(f"Starting optimization for {len(image_files)} images")

        if resume and self.output_file.exists():
            with self.output_file.open('r') as f:
                previous_results = json.load(f)
            for image_name, result_dict in previous_results.items():
                self.state.best_results[image_name] = ImageOptimizationResult(
                    parameters   = result_dict["best_parameters"],
                    text_results = result_dict["text_results"],
                    score        = result_dict["metrics"]["score"],
                    char_count   = result_dict["metrics"]["char_count"],
                    iterations   = result_dict["metrics"]["iterations"]
                )

        try:
            # Process parameter combinations
            for params in self.generate_parameter_combinations():
                self.process_combination(params, image_files)

                # Save progress periodically
                if self.state.iteration % self.save_frequency == 0:
                    self.save_results()

        except KeyboardInterrupt:
            logger.warning("Optimization interrupted by user")
        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}", exc_info = True)
            raise
        finally:
            # Save results one final time
            self.save_results()

        return self.state.best_results

# -------------------- Main Entry Point --------------------

if __name__ == "__main__":

    extractor = TextExtractor(headless = True)
    optimizer = ParameterOptimizer(extractor = extractor)

    try:
        image_files = extractor.find_image_files('images/books')
    except FileNotFoundError as e:
        logger.error(str(e))
        raise

    optimizer.optimize(image_files)
