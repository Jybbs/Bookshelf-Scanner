import json
import itertools
import numpy as np

from dataclasses     import dataclass, field
from pathlib         import Path
from ruamel.yaml     import YAML
from typing          import Any
from collections.abc import Iterator

from bookshelf_scanner import ModuleLogger, TextExtractor, Utils
logger = ModuleLogger('optimizer')()

# -------------------- Data Classes --------------------

@dataclass
class OptimizationResult:
    """
    Stores the best optimization results for a single image,
    including the parameters used and the resulting OCR text and metrics.
    """
    parameters   : dict[str, Any]          # Parameters that achieved best results
    text_results : list[tuple[str, float]] # OCR results for the image
    score        : float                   # Total character-weighted confidence score
    char_count   : int                     # Total number of characters recognized
    iterations   : int                     # Number of optimization iterations run

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
            self.parameters   = new_params
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
    best_results : dict[str, OptimizationResult] = field(default_factory = dict)
    iteration    : int = 0   # Current iteration number
    _modified    : bool = field(default = False, init = False)

    def update_result(
        self,
        image_name   : str,
        params       : dict[str, Any],
        ocr_results  : list[tuple[str, float]],
        iteration    : int
    ) -> bool:
        """
        Updates results for an image if the new score is better.

        Args:
            image_name  : Name of the image being processed
            params      : Parameter settings used
            ocr_results : OCR results for the image
            iteration   : Current iteration number

        Returns:
            True if results were updated, False otherwise
        """
        if image_name not in self.best_results:
            score, count = ParameterOptimizer.calculate_score(ocr_results)
            self.best_results[image_name] = OptimizationResult(
                parameters   = params,
                text_results = ocr_results,
                score        = score,
                char_count   = count,
                iterations   = iteration
            )
            self._modified = True
            return True

        if self.best_results[image_name].update_if_better(params, ocr_results, iteration):
            self._modified = True
            return True

        return False

    @property
    def modified(self) -> bool:
        """
        Checks if state has been modified since last access.

        Returns:
            True if state has been modified, False otherwise
        """
        was_modified = self._modified
        self._modified = False
        return was_modified

# -------------------- ParameterOptimizer Class --------------------

class ParameterOptimizer:
    """
    Optimizes the parameters for text extraction by testing various
    combinations and recording the best results.
    """
    PROJECT_ROOT = Utils.find_root('pyproject.toml')
    OUTPUT_FILE  = PROJECT_ROOT / 'bookshelf_scanner' / 'data' / 'results' / 'optimizer.json'
    PARAMS_FILE  = PROJECT_ROOT / 'bookshelf_scanner' / 'config' / 'params.yml'

    def __init__(
        self,
        extractor   : TextExtractor,
        output_file : Path | None = None,
        params_file : Path | None = None,
    ):
        """
        Initializes the ParameterOptimizer instance.

        Args:
            extractor   : An instance of TextExtractor to perform OCR
            output_file : Path to save optimization results
            params_file : Path to parameter configuration file
        """
        self.extractor   = extractor
        self.output_file = output_file or self.OUTPUT_FILE
        self.params_file = params_file or self.PARAMS_FILE

        self.state = OptimizerState()
        self.steps = self.extractor.initialize_processing_steps()
        logger.info("Initialized ParameterOptimizer.")

    def generate_parameter_combinations(self) -> Iterator[dict[str, Any]]:
        """
        Generates all possible parameter combinations for OCR processing.

        The function follows these rules:
        1. OCR is always enabled.
        2. For all other steps, we test every on/off combination.
        3. When a step is on, we test every possible value of its parameters.
        4. When a step is off, we don't include its parameters at all.

        Yields:
            Dictionary containing parameter combinations.
        """
        # First, build our parameter space for each step
        step_parameters = {}
        for step in self.steps:
            if step.name == 'ocr':
                continue  # OCR is always enabled
            params = {}
            for param in step.parameters:
                param_values = np.arange(param.min, param.max + param.step/2, param.step).tolist()
                params[param.name] = param_values
            step_parameters[step.name] = {
                'parameters': params
            }

        # Now generate all possible step on/off combinations (excluding OCR)
        step_names = list(step_parameters.keys())
        step_states = list(itertools.product([True, False], repeat=len(step_names)))

        for state_combo in step_states:
            params_override = {}
            # Set enabled/disabled states
            for step_name, is_enabled in zip(step_names, state_combo):
                params_override[step_name] = {'enabled': is_enabled}
                if is_enabled:
                    # For enabled steps, generate all combinations of parameters
                    step_params = step_parameters[step_name]['parameters']
                    param_names = list(step_params.keys())
                    param_values = list(step_params.values())
                    for values_combo in itertools.product(*param_values):
                        param_dict = dict(zip(param_names, values_combo))
                        params_override[step_name]['parameters'] = param_dict
                        yield params_override.copy()
                else:
                    yield params_override.copy()

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

        char_count  = sum(len(text.strip()) for text, _ in text_results)
        total_score = sum(len(text.strip()) * conf for text, conf in text_results)

        return total_score, char_count

    def save_results(self):
        """
        Saves current optimization results to the output file.
        """
        output_dict = {
            name: result.to_dict()
            for name, result in sorted(self.state.best_results.items())
        }

        with self.output_file.open('w', encoding='utf-8') as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=4)

        logger.info(f"Results saved to {self.output_file}")

    def optimize(self, image_files: list[Path]):
        """
        Runs the optimization process on the provided image files.

        Args:
            image_files : List of image file paths to process
        """
        if not image_files:
            raise ValueError("No image files provided")

        logger.info(f"Starting optimization for {len(image_files)} images")

        total_combinations = sum(1 for _ in self.generate_parameter_combinations())
        logger.info(f"Total parameter combinations to test: {total_combinations}")

        try:
            # Process parameter combinations
            for params_override in self.generate_parameter_combinations():
                self.state.iteration += 1

                # Set parameters in extractor
                self.extractor.initialize_processing_steps(params_override=params_override)

                # Perform OCR on images
                results = self.extractor.perform_ocr_headless(image_files)

                # Track improvements
                improvements = sum(
                    1 for image_name, ocr_results in results.items()
                    if self.state.update_result(
                        image_name  = image_name,
                        params       = params_override,
                        ocr_results  = ocr_results,
                        iteration    = self.state.iteration
                    )
                )

                # Only log progress periodically or when improvements occur
                if improvements > 0 or self.state.iteration % 100 == 0:
                    progress = (self.state.iteration / total_combinations) * 100
                    logger.info(
                        f"Progress: {progress:.1f}% - "
                        f"Combination {self.state.iteration:,}/{total_combinations:,} "
                        f"improved {improvements} images"
                    )

        except KeyboardInterrupt:
            logger.warning("Optimization interrupted by user")

        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}", exc_info=True)
            raise

        finally:
            if self.state.modified:
                self.save_results()

