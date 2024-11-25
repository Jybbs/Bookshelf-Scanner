import json
import logging
import itertools
import numpy as np

from dataclasses   import dataclass, field
from pathlib       import Path
from ruamel.yaml   import YAML
from TextExtractor import TextExtractor
from typing        import Any, Optional, Iterator

# -------------------- Configuration and Logging --------------------

logging.basicConfig(
    level    = logging.INFO,
    format   = '%(asctime)s - %(levelname)s - %(message)s',
    filename = Path(__file__).parent / 'ParameterOptimizer.log',
    filemode = 'w'
)
logger = logging.getLogger(__name__)

# -------------------- Data Classes --------------------

@dataclass
class OptimizationConfig:
    enabled_steps   : list[str]  # List of processing step names to optimize
    param_ranges    : dict[str, tuple[float, float, float]]  # Parameter ranges for optimization
    save_frequency  : int = 10   # How often to save intermediate results
    batch_size      : int = 100  # Number of combinations to process per batch

@dataclass
class ImageOptimizationResult:
    parameters   : dict[str, Any]           # Parameters that achieved best results
    text_results : list[tuple[str, float]]  # OCR results for the image
    score        : float                    # Total character-weighted confidence score
    char_count   : int                      # Total number of characters recognized
    iterations   : int = 0                  # Number of optimization iterations run

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
        Convert optimization result to dictionary format.
        
        Returns:
            Dictionary containing optimization results and metrics
        """
        return {
            "best_parameters": self.parameters,
            "text_results"   : self.text_results,
            "metrics": {
                "score"      : round(self.score, 4),
                "char_count" : self.char_count,
                "iterations" : self.iterations
            }
        }

@dataclass
class OptimizerState:
    best_results   : dict[str, ImageOptimizationResult] = field(default_factory = dict)  # Best results per image
    current_batch  : int = 0  # Index of current batch being processed
    total_batches  : int = 0  # Total number of parameter combination batches
    iteration      : int = 0  # Current iteration number

# -------------------- Main Optimizer Class --------------------optimi

class ParameterOptimizer:
    BATCH_SIZE     = 100
    OUTPUT_FILE    = Path(__file__).parent / 'optimized_results.json',
    PARAMS_FILE    = Path(__file__).resolve().parent.parent / 'config' / 'params.yml'
    SAVE_FREQUENCY = 10

    def __init__(
        self,
        extractor      : TextExtractor,
        batch_size     : int            = BATCH_SIZE,
        output_file    : Optional[Path] = None,
        params_file    : Optional[Path] = None,
        save_frequency : int            = SAVE_FREQUENCY
    ):
        """
        Initialize the parameter optimizer.
        
        Args:
            extractor      : Initialized TextExtractor instance
            batch_size     : Size of parameter combination batches
            output_file    : Path to save optimization results
            params_file    : Path to parameter configuration file
            save_frequency : How often to save intermediate results
        """
        self.extractor      = extractor
        self.batch_size     = batch_size
        self.output_file    = output_file or self.OUTPUT_FILE
        self.params_file    = params_file or self.PARAMS_FILE
        self.save_frequency = save_frequency
        
        # Initialize state
        self.state   = OptimizerState()
        self.config  = self.load_config()
        
        logger.info(f"Initialized optimizer with {len(self.config.enabled_steps)} enabled steps")

    # -------------------- Configuration Methods --------------------

    def load_config(self) -> OptimizationConfig:
        """
        Load and validate optimization configuration from params file.
        
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
        
        # Map step definitions to enabled steps in TextExtractor
        enabled_steps = []
        param_ranges  = {}
        
        for step_def in step_definitions:
            step_name = next(
                (step for step in self.extractor.steps 
                 if step.name == step_def['name'] and step.is_enabled),
                None
            )
            
            if step_name:
                enabled_steps.append(step_name.name)
                
                for param in step_def.get('parameters', []):
                    if all(k in param for k in ('name', 'min', 'max', 'step')):
                        param_ranges[param['name']] = (
                            float(param['min']),
                            float(param['max']),
                            float(param['step'])
                        )
        
        return OptimizationConfig(
            enabled_steps   = enabled_steps,
            param_ranges    = param_ranges,
            save_frequency  = self.save_frequency,
            batch_size      = self.batch_size
        )

    # -------------------- Parameter Generation --------------------

    @property
    def valid_combinations(self) -> Iterator[dict[str, Any]]:
        """
        Generate valid parameter combinations based on enabled steps.
        
        Yields:
            Dictionary containing a valid parameter combination
        """
        # Create mapping of parameters to their parent steps
        param_to_step  = {}
        enabled_params = {}
        
        for step in self.extractor.steps:
            if step.is_enabled:
                for param in step.parameters:
                    param_to_step[param.name] = step.name
                    enabled_params[param.name] = (
                        float(param.min),
                        float(param.max),
                        float(param.step)
                    )
        
        # Generate values for enabled parameters
        param_values = {
            name: np.arange(
                start = min_val,
                stop  = max_val + step / 2,
                step  = step
            ).tolist()
            for name, (min_val, max_val, step) in enabled_params.items()
        }
        
        # Generate combinations
        param_names = list(param_values.keys())
        for value_combo in itertools.product(*param_values.values()):
            combo = dict(zip(param_names, value_combo))
            
            # Add enabled flags for parent steps
            for step in self.extractor.steps:
                combo[f"use_{step.name}"] = step.is_enabled
            
            yield combo

    # -------------------- Scoring Methods --------------------

    @staticmethod
    def calculate_score(text_results: list[tuple[str, float]]) -> tuple[float, int]:
        """
        Calculate confidence-weighted character score for OCR results.
        
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
            char_weight  = 1.0
            total_score += text_chars * confidence * char_weight
            char_count  += text_chars
            
        return total_score, char_count

    # -------------------- Results Management --------------------

    def save_results(self) -> None:
        """Save current optimization results to the output file."""
        output_dict = {
            name: result.to_dict()
            for name, result in sorted(self.state.best_results.items())
        }
        
        with self.output_file.open('w', encoding='utf-8') as f:
            json.dump(output_dict, f, ensure_ascii = False, indent = 4)
        
        logger.info(f"Results saved to {self.output_file}")

    def process_combination(
        self,
        params      : dict[str, Any],
        image_files : list[Path]
    ) -> None:
        """
        Process a single parameter combination across all images.
        
        Args:
            params      : Parameter combination to test
            image_files : List of image files to process
        """
        self.state.iteration += 1
        
        # Run OCR with current parameters
        results = self.extractor.interactive_experiment(
            image_files     = image_files,
            params_override = params,
            output_json     = False,
            interactive_ui  = False
        )
        
        # Update best results for each image
        for image_name, ocr_results in results.items():
            if image_name not in self.state.best_results:
                score, count = self.calculate_score(ocr_results)
                self.state.best_results[image_name] = ImageOptimizationResult(
                    parameters   = params.copy(),
                    text_results = ocr_results,
                    score        = score,
                    char_count   = count
                )
            else:
                if self.state.best_results[image_name].update_if_better(
                    params,
                    ocr_results,
                    self.state.iteration
                ):
                    logger.info(
                        f"New best score for {image_name}: "
                        f"{self.state.best_results[image_name].score:.2f} "
                        f"({self.state.best_results[image_name].char_count} chars)"
                    )

    # -------------------- Main Optimization Method --------------------

    def optimize(
        self,
        image_files : list[Path],
        resume      : bool = False
    ) -> dict[str, ImageOptimizationResult]:
        """
        Run optimization process on provided image files.
        
        Args:
            image_files : List of image files to process
            resume      : Whether to resume from existing results
            
        Returns:
            Dictionary mapping image names to their optimization results
            
        Raises:
            ValueError: If no image files are provided
        """
        if not image_files:
            raise ValueError("No image files provided")
        
        # Load previous results if resuming
        if resume and self.output_file.exists():
            logger.info(f"Resuming optimization from {self.output_file}")
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
            for params in self.valid_combinations:
                self.process_combination(params, image_files)
                
                # Save progress periodically
                if self.state.iteration % self.save_frequency == 0:
                    self.save_results()
                    
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
        finally:
            self.save_results()
        
        return self.state.best_results

# -------------------- Main Entry Point --------------------

if __name__ == "__main__":
    # Initialize components
    extractor = TextExtractor(gpu_enabled = False)
    optimizer = ParameterOptimizer(extractor = extractor)
    
    # Find image files
    image_files = sorted(
        file for file in Path("images/books").glob("*.jpg")
        if file.is_file()
    )
    if not image_files:
        raise FileNotFoundError("No images found in images/books directory")
    
    # Run optimization
    optimizer.optimize(image_files)