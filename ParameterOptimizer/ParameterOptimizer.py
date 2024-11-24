import json
import logging
import itertools
import numpy as np

from dataclasses   import dataclass
from pathlib       import Path
from typing        import Any
from ruamel.yaml   import YAML
from TextExtractor import TextExtractor

# -------------------- Configuration and Logging --------------------

logging.basicConfig(
    level    = logging.INFO,
    format   = '%(asctime)s - %(levelname)s - %(message)s',
    filename = 'parameter_optimizer.log',
    filemode = 'w'
)
logger = logging.getLogger(__name__)

# -------------------- Data Classes --------------------

@dataclass
class ImageOptimizationResult:
    """
    Stores optimization results for a single image.
    
    Attributes:
        parameters  : Dictionary of parameters that achieved best results
        text_results: List of (text, confidence) tuples from OCR
        score       : Total character-weighted confidence score
        char_count  : Total number of characters recognized
    """
    parameters  : Dict[str, float]
    text_results: List[Tuple[str, float]]
    score       : float
    char_count  : int

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert optimization result to dictionary format matching project structure.
        
        Returns:
            Dictionary containing best parameters and OCR results
        """
        return {
            "text_results": self.text_results,
            "best_parameters": self.parameters,
            "metrics": {
                "score"     : round(self.score, 4),
                "char_count": self.char_count
            }
        }

# -------------------- Parameter Management --------------------

def load_parameter_ranges(params_file: Path) -> Dict[str, Tuple[float, float, float]]:
    """
    Extract parameter ranges from params.yml configuration.
    
    Args:
        params_file: Path to params.yml configuration file
        
    Returns:
        Dictionary mapping parameter names to (min, max, step) tuples
        
    Raises:
        FileNotFoundError: If params.yml cannot be found
    """
    yaml = YAML(typ='safe')
    
    try:
        with params_file.open('r') as f:
            config = yaml.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {params_file}")
        raise
        
    ranges = {}
    for step in config:
        for param in step.get('parameters', []):
            if all(k in param for k in ('name', 'min', 'max', 'step')):
                ranges[param['name']] = (
                    float(param['min']),
                    float(param['max']),
                    float(param['step'])
                )
    
    return ranges

def generate_parameter_combinations(
    param_ranges: Dict[str, Tuple[float, float, float]]
) -> List[Dict[str, float]]:
    """
    Generate all valid parameter combinations from defined ranges.
    
    Args:
        param_ranges: Dictionary of parameter ranges (min, max, step)
        
    Returns:
        List of parameter dictionaries for all valid combinations
    """
    param_values = {
        name: np.arange(
            start = min_val,
            stop  = max_val + step/2,  # Include max value
            step  = step
        )
        for name, (min_val, max_val, step) in param_ranges.items()
    }
    
    total_combinations = np.prod([len(values) for values in param_values.values()])
    logger.info(f"Generating {total_combinations} parameter combinations")
    
    return [
        dict(zip(param_values.keys(), values))
        for values in itertools.product(*param_values.values())
    ]

# -------------------- Optimization Logic --------------------

def calculate_score(text_results: List[Tuple[str, float]]) -> Tuple[float, int]:
    """
    Calculate confidence-weighted character score for OCR results.
    
    Score is the sum of (confidence * character_count) for each text segment.
    
    Args:
        text_results: List of (text, confidence) tuples from OCR
        
    Returns:
        Tuple of (total_score, total_character_count)
    """
    total_score = 0.0
    char_count  = 0
    
    for text, confidence in text_results:
        text_chars   = len(text)
        total_score += text_chars * confidence
        char_count  += text_chars
        
    return total_score, char_count

def optimize_parameters(
    text_extractor : Any,
    image_files    : List[Path],
    params_file    : Path,
    output_file    : Path,
    save_frequency : int = 10
) -> Dict[str, ImageOptimizationResult]:
    """
    Find optimal parameters for OCR processing of each image.
    
    Args:
        text_extractor : Initialized TextExtractor instance
        image_files    : List of image files to process
        params_file    : Path to params.yml configuration
        output_file    : Path to save optimization results
        save_frequency : How often to save intermediate results
        
    Returns:
        Dictionary mapping image names to their optimization results
    """
    # Load parameter space
    param_ranges = load_parameter_ranges(params_file)
    combinations = generate_parameter_combinations(param_ranges)
    
    # Track best results per image
    best_results: Dict[str, ImageOptimizationResult] = {}
    
    # Process all parameter combinations
    for combo_idx, params in enumerate(combinations, 1):
        logger.info(f"Testing parameter combination {combo_idx}/{len(combinations)}")
        
        # Run OCR with current parameters
        results = text_extractor.interactive_experiment(
            image_files     = image_files,
            params_override = params
        )
        
        # Update best results for each image
        for image_name, image_results in results.items():
            score, char_count = calculate_score(image_results)
            
            # Check if these parameters gave better results
            if (image_name not in best_results or 
                score > best_results[image_name].score):
                
                best_results[image_name] = ImageOptimizationResult(
                    parameters   = params.copy(),
                    text_results = image_results,
                    score       = score,
                    char_count  = char_count
                )
                
                logger.info(
                    f"New best score for {image_name}: "
                    f"{score:.2f} ({char_count} chars)"
                )
        
        # Save intermediate results periodically
        if combo_idx % save_frequency == 0:
            save_optimization_results(best_results, output_file)
    
    # Save final results
    save_optimization_results(best_results, output_file)
    return best_results

def save_optimization_results(
    results     : Dict[str, ImageOptimizationResult],
    output_file : Path
) -> None:
    """
    Save optimization results to JSON file.
    
    Args:
        results     : Dictionary of optimization results per image
        output_file : Path to save results
    """
    output_dict = {
        name: result.to_dict()
        for name, result in results.items()
    }
    
    with output_file.open('w') as f:
        json.dump(output_dict, f, indent=4, sort_keys=True)
    
    logger.info(f"Optimization results saved to {output_file}")

# -------------------- Main Entry Point --------------------

def main(
    image_dir   : str = "images/books",
    params_file : str = "params.yml",
    output_file : str = "optimized_ocr_results.json"
) -> None:
    """
    Main entry point for parameter optimization.
    
    Args:
        image_dir   : Directory containing images to process
        params_file : Path to parameter configuration file
        output_file : Path to save optimization results
    """
    # Initialize paths
    image_path   = Path(image_dir)
    params_path  = Path(params_file)
    output_path  = Path(output_file)
    
    # Find image files
    image_files = list(image_path.glob("*.jpg"))
    if not image_files:
        raise FileNotFoundError(f"No images found in {image_dir}")
    
    # Run optimization
    optimize_parameters(
        text_extractor = TextExtractor(),
        image_files    = image_files,
        params_file    = params_path,
        output_file    = output_path
    )

if __name__ == "__main__":
    main()