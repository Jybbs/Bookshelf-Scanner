import json
import numpy as np

from contextlib  import contextmanager
from dataclasses import dataclass
from functools   import cache
from itertools   import product
from pathlib     import Path
from typing      import Any, Iterator

from bookshelf_scanner import ModuleLogger, TextExtractor, Utils
logger = ModuleLogger('optimizer')()

# -------------------- Data Classes --------------------

@dataclass(frozen = True)
class ParameterRange:
    """
    Defines the valid range and step size for a processing parameter.
    """
    name     : str
    min_val  : float
    max_val  : float
    step_val : float
    value    : float

@dataclass(frozen = True)
class ProcessingStep:
    """
    Defines a processing step and its configurable parameters.
    """
    name       : str
    parameters : tuple[ParameterRange, ...]

@dataclass
class OCRResult:
    """
    Stores parameter evaluation results for a single image.
    """
    parameters   : dict[str, dict]
    ocr_results  : list[tuple[str, float]]
    score        : float
    improvements : dict[str, Any]

# -------------------- ParameterOptimizer Class --------------------

class ParameterOptimizer:
    """
    Tests parameter combinations to find optimal OCR configurations.
    Evaluates each combination against individual images to maximize
    character-level confidence scores.
    """
    
    # -------------------- Class Constants --------------------
    
    PROJECT_ROOT = Utils.find_root('pyproject.toml')
    RESULTS_DIR  = PROJECT_ROOT / 'bookshelf_scanner' / 'data' / 'results'
    OUTPUT_FILE  = RESULTS_DIR / 'optimizer.json'
    
    def __init__(self, extractor: TextExtractor):
        """
        Initialize the parameter optimizer.
        
        Args:
            extractor : TextExtractor instance to optimize
        """
        self.extractor = extractor
        self.results   = {}

    @contextmanager
    def graceful_exit(self):
        """
        Context manager to ensure results are saved even if optimization is interrupted.
        Catches KeyboardInterrupt and saves current results before re-raising.
        """
        try:
            yield

        except KeyboardInterrupt:
            logger.info("\nOptimization interrupted. Saving current results...")
            self.save_results()
            raise

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            self.save_results()
            raise
    
    # -------------------- Parameter Space Generation --------------------
    
    def extract_step_definitions(self) -> list[ProcessingStep]:
        """
        Creates immutable processing step definitions from extractor configuration.
        
        Returns:
            List of ProcessingStep instances containing parameter ranges
        """
        steps           = self.extractor.initialize_processing_steps()
        processing_steps = []
        
        for step in steps:
            if step.parameters and step.is_pipeline:
                param_ranges = tuple(
                    ParameterRange(
                        name     = p.name,
                        min_val  = float(p.min),
                        max_val  = float(p.max),
                        step_val = float(p.step),
                        value    = float(p.value)
                    )
                    for p in step.parameters
                )
                processing_steps.append(ProcessingStep(name = step.name, parameters = param_ranges))
        
        return processing_steps
    
    @cache
    def map_step_variations(self, step: ProcessingStep) -> list[dict[str, float]]:
        """
        Maps all valid parameter variations for a single processing step.
        Ensures parameter values maintain 1 decimal place precision.
        
        Args:
            step : ProcessingStep containing parameter definitions
            
        Returns:
            List of parameter value combinations for the step
        """
        param_ranges = [
            (param.name, [
                int(v) if isinstance(param.value, int) else round(v, 1)
                for v in np.arange(param.min_val, param.max_val + param.step_val, param.step_val)
            ])
            for param in step.parameters
        ]
        
        parameter_variations = [{
            'parameters' : {
                name: value for name, value in zip(
                    [name for name, _ in param_ranges],
                    value_combo
                )
            },
            'enabled'    : True
        } for value_combo in product(*(values for _, values in param_ranges))]
        
        return parameter_variations
    
    def build_parameter_grid(self) -> Iterator[dict[str, dict]]:
        """
        Builds complete grid of parameter combinations across processing steps.
        
        Returns:
            Iterator of complete parameter configurations to test
        """
        processing_steps   = self.extract_step_definitions()
        step_variations    = []
        total_combinations = 1
        
        # Calculate variations per step and total
        for step in processing_steps:
            step_params = self.map_step_variations(step)
            step_variations.append((step.name, step_params))
            total_combinations *= len(step_params)
            
        logger.info(f"Generated {total_combinations} parameter combinations to test")
        
        # Generate each complete combination
        parameter_grid = (
            {
                name: params
                for name, params in zip([s[0] for s in step_variations], step_combo)
            }
            for step_combo in product(*[params for _, params in step_variations])
        )
        
        return parameter_grid
    
    # -------------------- Parameter Evaluation --------------------
    
    def identify_improvements(
        self,
        new_params : dict,
        old_params : dict
    ) -> dict:
        """
        Identifies which parameters changed from the previous best configuration.
        
        Args:
            new_params : New parameter configuration
            old_params : Previous best parameter configuration
            
        Returns:
            Dictionary of parameter improvements
        """
        improvements = {}
        
        for step_name, step_config in new_params.items():
            if step_name not in old_params:
                improvements[step_name] = step_config
                continue
            
            old_config = old_params[step_name]
            if step_config.get('enabled') != old_config.get('enabled'):
                improvements[step_name] = {'enabled': step_config.get('enabled')}
                continue
            
            new_values = step_config.get('parameters', {})
            old_values = old_config.get('parameters', {})
            
            changed_values = {
                key: value
                for key, value in new_values.items()
                if key not in old_values or old_values[key] != value
            }
            
            if changed_values:
                improvements[step_name] = {'parameters': changed_values}
        
        return improvements
    
    # -------------------- Primary Methods --------------------
    
    def save_results(self):
        """
        Saves current optimization results to JSON file.
        Creates output directory if it doesn't exist.
        """
        output = {
            str(Path(path).name): {
                'parameters'   : result.parameters,
                'texts'        : result.ocr_results,
                'score'        : result.score,
                'char_count'   : sum(len(text) for text, _ in result.ocr_results),
                'improvements' : result.improvements
            }
            for path, result in self.results.items()
        }
        
        self.OUTPUT_FILE.parent.mkdir(exist_ok = True, parents = True)
        with self.OUTPUT_FILE.open('w') as f:
            json.dump(output, f, indent = 2)
            
        logger.info(f"Results saved to {self.OUTPUT_FILE}")

    def evaluate_image(self, image_path: Path) -> OCRResult:
        """
        Tests parameter combinations systematically to find the highest scoring configuration.
        
        Args:
            image_path : Path to the image being evaluated
            
        Returns:
            OCRResult containing best parameters and scores
        """
        logger.info(f"Evaluating parameters for {image_path.name}")
        best_score  = 0.0
        best_params = None
        best_ocr    = []
        
        for current, params in enumerate(self.build_parameter_grid(), 1):

            # Initialize extractor with current parameter combination
            self.extractor.initialize_processing_steps(params_override = params)
            
            # Process image and calculate score
            results     = self.extractor.perform_ocr_headless([image_path])
            ocr_results = results.get(image_path.name, [])
            score       = round(sum(len(text) * conf for text, conf in ocr_results), 3)
            
            # Log progress at intervals
            if current % 100000 == 0:
                logger.info(f"Testing combination {current}")
            
            # Update if score improved
            if score > best_score:
                improvements = {}
                if best_params:
                    improvements = self.identify_improvements(params, best_params)
                    logger.info(f"Parameter improvements: {json.dumps(improvements, indent = 2)}")
                
                best_score  = score
                best_params = params
                best_ocr    = ocr_results
                
                logger.info(f"New best score {score:.3f} on combination {current}")
        
        # Store final results
        result = OCRResult(
            parameters   = best_params,
            ocr_results  = best_ocr,
            score        = best_score,
            improvements = improvements
        )
        
        self.results[str(image_path)] = result
        return result
    
    def optimize(self, image_files: list[Path]):
        """
        Searches for optimal parameters across all provided images.
        Saves comprehensive results including parameters, OCR outputs, and scores.
        Handles interruptions gracefully by saving partial results.
        
        Args:
            image_files : List of images to evaluate
        """
        logger.info(f"Beginning parameter search for {len(image_files)} images")
        
        with self.graceful_exit():
            for image_path in image_files:
                self.evaluate_image(image_path)
            
            self.save_results()
            
        logger.info("Parameter optimization complete")