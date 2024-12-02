import json
import logging
import itertools
import numpy as np

from dataclasses     import dataclass, field
from functools       import cached_property
from pathlib         import Path
from ruamel.yaml     import YAML
from typing          import Any, Iterator
from collections.abc import Sequence

from bookshelf_scanner import TextExtractor

# -------------------- Configuration and Logging --------------------

logger = logging.getLogger('ParameterOptimizer')
logger.setLevel(logging.INFO)

handler = logging.FileHandler(Path(__file__).parent / 'optimizer.log', mode = 'w')
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
    batch_size     : int = 100                              # Number of combinations to process per batch

    @cached_property
    def total_combinations(self) -> int:
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
            # Store enabled steps and their associated parameter values
            filtered_params = {}
            
            # First add all the enabled/disabled flags
            for key, value in new_params.items():
                if key.startswith('use_'):
                    filtered_params[key] = value
            
            # Then add parameter values for enabled steps
            for key, value in new_params.items():
                if not key.startswith('use_'):
                    # Extract step name from parameter (e.g., 'shadow_kernel_size' -> 'shadow_removal')
                    step_name = key.split('_')[0]
                    use_key = f'use_{step_name}_removal' if 'removal' in key else f'use_{step_name}'
                    
                    # If this parameter's step is enabled, include it
                    if new_params.get(use_key, False):
                        filtered_params[key] = value

            self.parameters   = filtered_params
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
    best_results : dict[str, ImageOptimizationResult] = field(default_factory = dict)
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
            params     : Parameter settings used
            ocr_results: OCR results for the image
            iteration  : Current iteration number

        Returns:
            True if results were updated, False otherwise
        """
        if image_name not in self.best_results:
            score, count = ParameterOptimizer.calculate_score(ocr_results)
            self.best_results[image_name] = ImageOptimizationResult(
                parameters   = {k: v for k, v in params.items() if k.startswith('use_') and v or not k.startswith('use_')},
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
    BATCH_SIZE  : int  = 100
    OUTPUT_FILE : Path = Path(__file__).parent / 'optimized_results.json'
    PARAMS_FILE : Path = Path(__file__).resolve().parent.parent.parent / 'config' / 'params.yml'

    def __init__(
        self,
        extractor   : TextExtractor,
        batch_size  : int         = BATCH_SIZE,
        output_file : Path | None = None,
        params_file : Path | None = None,
    ):
        """
        Initializes the ParameterOptimizer instance.

        Args:
            extractor   : An instance of TextExtractor to perform OCR
            batch_size  : Number of parameter combinations to process per batch
            output_file : Path to save optimization results
            params_file : Path to parameter configuration file
        """
        self.extractor   = extractor
        self.batch_size  = batch_size
        self.output_file = output_file or self.OUTPUT_FILE
        self.params_file = params_file or self.PARAMS_FILE

        self.state  = OptimizerState()
        self.config = self.load_config()

        logger.info(f"Initialized with {self.config.total_combinations:,} parameter combinations to test")

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
            enabled_steps = enabled_steps,
            param_ranges  = param_ranges,
            batch_size    = self.batch_size
        )

    def generate_parameter_combinations(self) -> Iterator[dict[str, Any]]:
        """
        Generates all possible parameter combinations for OCR processing.
        
        The function follows these rules:
        1. OCR is always enabled
        2. For all other steps, we test every on/off combination 
        3. When a step is on, we test every possible value of its parameters
        4. When a step is off, we don't include its parameters at all

        Yields:
            Dictionary containing parameter combinations in the format:
            {
                'use_ocr': True,
                'use_step_name': bool,  # For each step
                'parameter_name': value  # Only for enabled steps
            }
        """
        # First, build our parameter space for each step
        step_parameters = {}
        
        for step_name in self.config.enabled_steps:
            # Get all parameters belonging to this step
            base_name = step_name.replace('_removal', '')
            params = {}
            
            for param_name, (min_val, max_val, step_size) in self.config.param_ranges.items():
                if param_name.startswith(base_name):
                    params[param_name] = np.arange(min_val, max_val + step_size/2, step_size).tolist()
            
            if params:
                step_parameters[step_name] = params
        
        # Now generate all possible step on/off combinations (except OCR)
        step_states = list(itertools.product([True, False], repeat=len(self.config.enabled_steps)))
        
        # For each possible combination of enabled/disabled steps...
        for state_combo in step_states:
            step_states_dict = {'use_ocr': True}
            enabled_params = {}
            
            # Map states to steps and collect parameters for enabled steps
            for step_name, is_enabled in zip(self.config.enabled_steps, state_combo):
                step_states_dict[f'use_{step_name}'] = is_enabled
                
                if is_enabled and step_name in step_parameters:
                    enabled_params.update(step_parameters[step_name])
            
            if not enabled_params:
                # If no steps are enabled (except OCR), yield just the step states
                yield step_states_dict
                continue
                
            # Generate all possible parameter value combinations for enabled steps
            param_names = list(enabled_params.keys())
            param_values = list(enabled_params.values())
            
            for value_combo in itertools.product(*param_values):
                result = step_states_dict.copy()
                result.update(dict(zip(param_names, value_combo)))
                yield result

    @staticmethod
    def calculate_score(text_results: Sequence[tuple[str, float]]) -> tuple[float, int]:
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

        with self.output_file.open('w', encoding = 'utf-8') as f:
            json.dump(output_dict, f, ensure_ascii = False, indent = 4)

        if self.state.iteration >= self.config.total_combinations:
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
    ) -> int:
        """
        Processes a single parameter combination across all images.

        Args:
            params      : Dictionary of parameter settings to test
            image_files : List of image file paths to process

        Returns:
            Number of images that showed improvement
        """
        self.state.iteration += 1

        # Set parameters in extractor
        self.extractor.initialize_steps(params_override = params)

        # Extract text using extractor's headless method
        results = self.extractor.extract_text_headless(image_files)

        # Track improvements
        improvements = sum(
            1 for image_name, ocr_results in results.items()
            if self.state.update_result(
                image_name  = image_name,
                params     = params,
                ocr_results = ocr_results,
                iteration  = self.state.iteration
            )
        )

        # Only log progress periodically or when improvements occur
        if improvements > 0 or self.state.iteration % 100 == 0:
            progress = (self.state.iteration / self.config.total_combinations) * 100
            logger.info(
                f"Progress: {progress:.1f}% - "
                f"Combination {self.state.iteration:,}/{self.config.total_combinations:,} "
                f"improved {improvements} images"
            )

        return improvements

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

        except KeyboardInterrupt:
            logger.warning("Optimization interrupted by user")

        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}", exc_info = True)
            raise

        finally:
            # Only save results at the end or if there were improvements
            if self.state.modified:
                self.save_results()

        return self.state.best_results