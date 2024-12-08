import json
import numpy    as np
import torch
import torch.nn as nn

from collections         import defaultdict
from dataclasses         import dataclass
from pathlib             import Path
from torch.nn.functional import mse_loss
from torch.utils.data    import DataLoader, random_split, TensorDataset
from typing              import Any

from bookshelf_scanner   import ModuleLogger, TextExtractor, Utils
logger = ModuleLogger('optimizer')()

# -------------------- Neural Network Components --------------------

class ParameterEncoder(nn.Module):
    """
    Encodes parameter vectors into latent representations for performance prediction.
    
    This encoder reduces the dimensionality of parameter vectors, capturing essential features
    that influence OCR performance. The architecture consists of two linear layers with ReLU
    activations and dropout for regularization.
    """
    def __init__(self, input_dimension: int, latent_dimension: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, latent_dimension),
            nn.ReLU()
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.network(input_tensor)

class PerformancePredictor(nn.Module):
    """
    Predicts OCR performance from latent parameter representations.
    
    This predictor estimates the OCR performance score based on latent embeddings produced
    by the ParameterEncoder. It uses a series of linear layers with ReLU activations and
    dropout, culminating in an output layer without activation to allow unrestricted score ranges.
    """
    def __init__(self, latent_dimension: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dimension, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
    def forward(self, latent_tensor: torch.Tensor) -> torch.Tensor:
        return self.network(latent_tensor)

class MetaLearningModel(nn.Module):
    """
    Combines ParameterEncoder and PerformancePredictor for meta-learning.
    
    This model encodes parameter configurations into latent vectors and predicts their
    corresponding OCR performance scores. The modular design allows for separate training
    and fine-tuning of the encoder and predictor components.
    """
    def __init__(self, input_dimension: int, latent_dimension: int = 64):
        super().__init__()
        self.parameter_encoder     = ParameterEncoder(input_dimension, latent_dimension)
        self.performance_predictor = PerformancePredictor(latent_dimension)
        
    def forward(self, input_tensor: torch.Tensor):
        latent_representation = self.parameter_encoder(input_tensor)
        performance_score     = self.performance_predictor(latent_representation)
        return latent_representation, performance_score

# -------------------- Data Classes --------------------

@dataclass
class ClusterMember:
    """
    Represents a member within a parameter cluster.
    
    Attributes:
        parameter_vector  : The parameter configuration vector.
        performance_score : The OCR performance score achieved with this configuration.
        latent_vector     : The latent representation of the parameter vector.
    """
    parameter_vector  : torch.Tensor
    performance_score : float
    latent_vector     : torch.Tensor

    @classmethod
    def from_dict(cls, data: dict) -> 'ClusterMember':
        """
        Creates a ClusterMember instance from a dictionary.

        Args:
            data: Dictionary containing 'parameters', 'score', and 'latent' keys.

        Returns:
            ClusterMember instance.
        """
        return cls(
            parameter_vector  = torch.tensor(data['parameters'], dtype = torch.float32),
            performance_score = data['score'],
            latent_vector     = torch.tensor(data['latent'], dtype = torch.float32)
        )

@dataclass
class OCRResult:
    """
    Represents a single OCR (Optical Character Recognition) result.
    
    Attributes:
        text       : The extracted text string from the image.
        confidence : The confidence score associated with the extracted text.
    """
    text       : str
    confidence : float

    def to_dict(self) -> dict:
        """
        Converts the OCRResult instance into a dictionary suitable for JSON serialization.
        """
        return {
            'text'       : self.text,
            'confidence' : self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'OCRResult':
        """
        Creates an OCRResult instance from a dictionary.

        Args:
            data: A dictionary containing 'text' and 'confidence' keys.

        Returns:
            OCRResult: An instance of OCRResult populated with the provided data.
        """
        return cls(text = data['text'], confidence = data['confidence'])
    
    @classmethod
    def from_tuples(cls, ocr_tuples: list[tuple[str, float]]) -> list['OCRResult']:
        """
        Creates a list of OCRResult instances from a list of (text, confidence) tuples.
        
        Args:
            ocr_tuples: List of tuples containing (text, confidence).
            
        Returns:
            List of OCRResult instances.
        """
        return [cls(text = text, confidence = confidence) for text, confidence in ocr_tuples]

@dataclass
class OptimizationRecord:
    """
    Records the outcome of a single optimization attempt for an image.
    
    Attributes:
        image_path    : Path to the image being optimized.
        parameters    : Best-found parameter vector (torch.Tensor).
        score         : OCR performance score achieved (float).
        latent_vector : Latent representation of the parameters (torch.Tensor).
        ocr_results   : List of OCRResult instances containing extracted text and confidence.
    """
    image_path    : Path
    parameters    : torch.Tensor
    score         : float
    latent_vector : torch.Tensor
    ocr_results   : list[OCRResult]

    def to_dict(self) -> dict:
        """
        Converts the OptimizationRecord instance into a dictionary suitable for JSON serialization.
        
        Returns:
            dict: A dictionary containing image path, parameters, score, latent vector, and OCR results.
        """
        return {
            'image_path'    : str(self.image_path),
            'parameters'    : self.parameters.tolist(),
            'score'         : self.score,
            'latent_vector' : self.latent_vector.tolist(),
            'ocr_results'   : [ocr.to_dict() for ocr in self.ocr_results]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'OptimizationRecord':
        """
        Creates an OptimizationRecord instance from a dictionary.

        Args:
            data : A dictionary containing 'image_path', 'parameters', 'score', 'latent_vector', and 'ocr_results'.

        Returns:
            OptimizationRecord: An instance of OptimizationRecord populated with the provided data.
        """
        return cls(
            image_path    = Path(data['image_path']),
            parameters    = torch.tensor(data['parameters'], dtype = torch.float32),
            score         = data['score'],
            latent_vector = torch.tensor(data['latent_vector'], dtype = torch.float32),
            ocr_results   = [OCRResult.from_dict(ocr) for ocr in data.get('ocr_results', [])]
        )
    
    def update_if_better(self, other: 'OptimizationRecord') -> bool:
        """
        Updates the current record with another record if the other has a better score.

        Args:
            other: Another OptimizationRecord to compare with.

        Returns:
            bool: True if the current record was updated, False otherwise.
        """
        if other.score > self.score:
            self.parameters    = other.parameters
            self.score         = other.score
            self.latent_vector = other.latent_vector
            self.ocr_results   = other.ocr_results
            return True
        return False

@dataclass
class MetaLearningState:
    """
    Maintains the state of the meta-learning optimizer.
    
    Attributes:
        model                : Instance of MetaLearningModel.
        optimization_history : List of OptimizationRecords.
        parameter_clusters   : Clusters of similar latent vectors.
        score_scaling        : Tuple tracking (min, max) scores encountered.
    """
    model                : MetaLearningModel
    optimization_history : list[OptimizationRecord]  = None
    parameter_clusters   : dict[int, dict[str, Any]] = None
    score_scaling        : tuple[float, float]        = (0.0, 1.0)

    def __post_init__(self):
        if self.optimization_history is None:
            self.optimization_history = []

        if self.parameter_clusters is None:
            self.parameter_clusters = defaultdict(lambda: {'members': [], 'center': None})
        
    def update_scaling(self, new_score: float):
        """
        Updates the score scaling based on a new score.

        Args:
            new_score: New OCR performance score.
        """
        minimum_score, maximum_score = self.score_scaling
        self.score_scaling = (
            min(minimum_score, new_score),
            max(maximum_score, new_score)
        )

# -------------------- Class Constants --------------------

class ParameterOptimizer:
    """
    Coordinates meta-learning guided parameter optimization for OCR.
    
    This optimizer leverages historical optimization data to suggest and refine
    parameter configurations that maximize OCR performance. It manages the meta-learning
    model, clusters parameter configurations, and handles training and persistence.

    Attributes:
        extractor               : Instance of TextExtractor being optimized.
        device_type             : Computation device ('cpu' or 'cuda').
        initial_points_count    : Number of initial parameter suggestions.
        iteration_count         : Number of refinement iterations per image.
        training_batch_size     : Batch size for training the meta-learning model.
        learning_rate_value     : Learning rate for the optimizer.
        parameter_boundaries    : Definitions of parameter ranges.
        optimizer_state         : Instance of MetaLearningState.
        model_optimizer         : Optimizer for training the meta-learning model.
        learning_rate_scheduler : Learning rate scheduler.
        optimization_results    : Dictionary to store optimization results.
    """

    # -------------------- Project & File Directories --------------------
    PROJECT_ROOT_DIRECTORY = Utils.find_root('pyproject.toml')
    RESULTS_DIRECTORY      = PROJECT_ROOT_DIRECTORY / 'bookshelf_scanner' / 'data' / 'results'
    MODEL_DIRECTORY        = PROJECT_ROOT_DIRECTORY / 'bookshelf_scanner' / 'core' / 'parameter_optimizer' / 'models'
    OUTPUT_JSON_FILE       = RESULTS_DIRECTORY / 'optimizer.json'
    MODEL_PYTORCH_FILE     = MODEL_DIRECTORY / 'meta_learner.pt'

    # -------------------- Initialization & Setup --------------------
    
    def __init__(
        self,
        extractor                  : TextExtractor,
        cluster_distance_threshold : float = 2.0,
        device_type                : str   = 'cuda' if torch.cuda.is_available() else 'cpu',
        initial_points_count       : int   = 10,
        iteration_count            : int   = 40,
        learning_rate_value        : float = 1e-3,
        training_batch_size        : int   = 16,
        training_buffer_size       : int   = 30,
        ucb_beta                   : float = 0.1,
    ):
        """
        Initializes the MetaLearningOptimizer.

        Args:
            extractor                  : TextExtractor instance to optimize (required).
            cluster_distance_threshold : Distance threshold for clustering.
            device_type                : Device for computations ('cpu' or 'cuda').
            initial_points_count       : Number of initial parameter suggestions.
            iteration_count            : Number of refinement steps per image.
            learning_rate_value        : Learning rate for the optimizer.
            training_batch_size        : Batch size for training.
            training_buffer_size       : Number of samples to accumulate before training the model.
            ucb_beta                   : Exploration parameter for Upper Confidence Bound (UCB) acquisition.
        """
        self.extractor                  = extractor
        self.cluster_distance_threshold = cluster_distance_threshold
        self.device_type                = device_type
        self.initial_points_count       = initial_points_count
        self.iteration_count            = iteration_count
        self.learning_rate_value        = learning_rate_value
        self.training_batch_size        = training_batch_size
        self.training_buffer_size       = training_buffer_size
        self.ucb_beta                   = ucb_beta

        # Ensure extractor config_state is initialized
        self.extractor.initialize_processing_steps()
        
        # Extract parameter boundaries from the extractor configuration
        self.parameter_boundaries = self.extract_parameter_boundaries()
        
        input_dimension      = len(self.parameter_boundaries)
        self.optimizer_state = MetaLearningState(
            model = self.initialize_meta_learning_model(input_dimension)
        )
        
        # Initialize the meta-learner optimizer and scheduler
        self.model_optimizer = torch.optim.Adam(
            self.optimizer_state.model.parameters(), 
            lr = self.learning_rate_value
        )
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.model_optimizer, 
            factor   = 0.5, 
            patience = 5
        )
        
        self.optimization_results = {}
        
        # Load existing state if available
        self.load_optimizer_state()
        
        # Initialize training buffer for batching training samples
        self.training_buffer = []

    def initialize_meta_learning_model(self, input_dimension: int) -> MetaLearningModel:
        """
        Initializes the MetaLearningModel with the given input dimension.

        Args:
            input_dimension: The dimension of the input parameter vectors.

        Returns:
            An instance of MetaLearningModel.
        """
        return MetaLearningModel(input_dimension).to(self.device_type)

    # -------------------- Parameter Space & Conversion --------------------

    def extract_parameter_boundaries(self) -> list[dict[str, Any]]:
        """
        Extracts parameter range definitions from the extractor's configuration.
        
        Returns:
            List of dictionaries defining each parameter's bounds, step, and type.
        """
        return [
            {
                'name'       : f"{step_name}.{param_name}",
                'min_value'  : float(param_definition["min"]),
                'max_value'  : float(param_definition["max"]),
                'step_value' : float(param_definition["step"]),
                'is_integer' : isinstance(param_definition["value"], int)
            }
            for step_name, step_definition in self.extractor.config_state.config_dict["steps"].items()
            if step_definition.get("parameters") is not None
            for param_name, param_definition in step_definition["parameters"].items()
        ]
    
    def vector_to_parameter_dictionary(self, parameter_vector: torch.Tensor) -> dict[str, Any]:
        """
        Converts a normalized parameter vector to a parameter dictionary.
        
        Scales each value from [0,1] to its original range and rounds if necessary.
        This dictionary can then be applied as a config override for the extractor.
        
        Args:
            parameter_vector: Normalized parameter vector (torch.Tensor).

        Returns:
            Dictionary structured for the extractor's config override.
        """
        parameter_dictionary = {"steps": {}}
        vector_numpy = parameter_vector.cpu().numpy()

        names      = [b['name']       for b in self.parameter_boundaries]
        min_values = np.array([b['min_value']  for b in self.parameter_boundaries])
        max_values = np.array([b['max_value']  for b in self.parameter_boundaries])
        is_integer = np.array([b['is_integer'] for b in self.parameter_boundaries])

        scaled_values = vector_numpy * (max_values - min_values) + min_values
        scaled_values = np.where(is_integer, np.round(scaled_values).astype(int), np.round(scaled_values, 1))

        for name, val in zip(names, scaled_values):
            # Insert parameter values back into the hierarchical step structure
            step_name, param_name = name.split('.')
            if step_name not in parameter_dictionary["steps"]:
                parameter_dictionary["steps"][step_name] = {
                    'enabled'    : True,
                    'parameters' : {}
                }
            parameter_dictionary["steps"][step_name]['parameters'][param_name] = {
                "value": int(val) if isinstance(val, np.integer) else float(val)
            }

        return parameter_dictionary
    
    # -------------------- Persistence & State Management --------------------

    def load_optimizer_state(self):
        """
        Loads the meta-learning state from disk if available.
        
        This restores the model's weights, optimization history (including OCR results),
        parameter clusters, and score scaling. After loading, it recalculates cluster centers
        to ensure no cluster center is None.
        """
        if self.MODEL_PYTORCH_FILE.exists():
            checkpoint = torch.load(self.MODEL_PYTORCH_FILE, map_location = self.device_type, weights_only = True)

            # Load model state
            self.optimizer_state.model.load_state_dict(checkpoint['model_state_dict'])

            # Restore optimization history
            self.optimizer_state.optimization_history = [
                OptimizationRecord.from_dict(record)
                for record in checkpoint['optimization_history']
            ]

            # Restore parameter clusters
            parameter_clusters = defaultdict(lambda: {'members': [], 'center': None})
            for cluster_id_str, members in checkpoint['parameter_clusters'].items():
                cluster_id_int = int(cluster_id_str)
                cluster_members = [ClusterMember.from_dict(m) for m in members]
                parameter_clusters[cluster_id_int]['members'].extend(cluster_members)

            # Recalculate cluster centers now that we have members
            for _, cinfo in parameter_clusters.items():
                if cinfo['members']:
                    # Compute the mean latent vector of all members to serve as the cluster center
                    latent_vectors = torch.stack([m.latent_vector for m in cinfo['members']])
                    cinfo['center'] = latent_vectors.mean(dim = 0)
                else:
                    # If no members, center remains None
                    cinfo['center'] = None

            self.optimizer_state.parameter_clusters = parameter_clusters

            # Restore score scaling
            self.optimizer_state.score_scaling = tuple(checkpoint['score_scaling'])
            
            logger.info(f"State loaded from {self.MODEL_PYTORCH_FILE}")

    def save_optimizer_state(self):
        """
        Saves the current meta-learning state and model to disk.
        
        Includes model weights, optimization history (with OCR results), parameter clusters, and score scaling.
        """
        self.MODEL_PYTORCH_FILE.parent.mkdir(exist_ok = True, parents = True)
        
        checkpoint_data = {
            'model_state_dict' : self.optimizer_state.model.state_dict(),
            'optimization_history' : [record.to_dict() for record in self.optimizer_state.optimization_history],
            'parameter_clusters' : {
                cluster_id: [
                    {
                        'parameters' : member.parameter_vector.tolist(),
                        'score'      : member.performance_score,
                        'latent'     : member.latent_vector.tolist()
                    } for member in cluster_info['members']
                ] for cluster_id, cluster_info in self.optimizer_state.parameter_clusters.items()
            },
            'score_scaling' : self.optimizer_state.score_scaling
        }
        
        torch.save(checkpoint_data, self.MODEL_PYTORCH_FILE)
        logger.info(f"Model and state saved to {self.MODEL_PYTORCH_FILE}")

    def save_optimization_results(self):
        """
        Saves the optimization results to a JSON file.
        
        Records the best parameters, scores, and OCR results for each processed image,
        allowing external analysis or future reference.
        """
        output_data = {
            str(record.image_path): record.to_dict()
            for record in self.optimizer_state.optimization_history
        }
        
        self.OUTPUT_JSON_FILE.parent.mkdir(exist_ok = True, parents = True)
        
        with self.OUTPUT_JSON_FILE.open('w', encoding = 'utf-8') as output_file:
            json.dump(output_data, output_file, indent = 2, ensure_ascii = False)
        
        logger.info(f"Results saved to {self.OUTPUT_JSON_FILE}")

    # -------------------- Parameter Suggestion & Clustering --------------------

    @torch.inference_mode()
    def suggest_initial_parameters(self) -> list[torch.Tensor]:
        """
        Suggests initial parameter configurations for evaluation.
        
        Combines best parameters from existing clusters with random samples to balance
        exploration (new random points) and exploitation (best cluster parameters).
        
        Returns:
            List of parameter vectors (torch.Tensor) to evaluate.
        """
        suggested_parameters = []

        # Add best parameters from existing clusters
        for cluster_info in self.optimizer_state.parameter_clusters.values():
            members = cluster_info['members']
            if members:
                best_member = max(members, key=lambda member: member.performance_score)
                suggested_parameters.append(best_member.parameter_vector)

        # If existing suggestions are available, diversify around the best found parameter
        if suggested_parameters:
            best_parameter_vector = suggested_parameters[0]
            needed = self.initial_points_count - len(suggested_parameters)
            for _ in range(needed):
                noise = torch.randn_like(best_parameter_vector) * 0.1
                suggested_parameters.append(torch.clamp(best_parameter_vector + noise, 0, 1))
        else:
            # If no existing suggestions, generate purely random parameter vectors
            suggested_parameters.extend(
                torch.rand(len(self.parameter_boundaries)) for _ in range(self.initial_points_count)
            )

        return suggested_parameters

    @torch.inference_mode()
    def update_parameter_clusters(
        self, 
        parameter_vector  : torch.Tensor, 
        performance_score : float
    ):
        """
        Assigns parameter configurations to latent clusters in the learned latent space.
        
        Encodes parameters to latent space, then assigns them to the nearest cluster or
        creates a new cluster if no close match exists. This helps structure the explored
        parameter space, aiding future suggestions.
        
        Args:
            parameter_vector  : Parameter configuration vector.
            performance_score : OCR performance score achieved with this configuration.
        """
        self.optimizer_state.model.eval()

        # Encode the parameter vector to latent space for clustering
        latent_representation, _ = self.optimizer_state.model(parameter_vector.unsqueeze(0))
        latent_representation = latent_representation.squeeze().cpu()

        # If no clusters exist, initialize the first cluster
        if not self.optimizer_state.parameter_clusters:
            self.optimizer_state.parameter_clusters[0]['members'].append(
                ClusterMember(parameter_vector, performance_score, latent_representation)
            )
            self.optimizer_state.parameter_clusters[0]['center'] = latent_representation
            return
        
        # Compute distances to existing cluster centers
        cluster_ids     = list(self.optimizer_state.parameter_clusters.keys())
        cluster_centers = torch.stack([
            cluster_info['center'] 
            for cluster_info in self.optimizer_state.parameter_clusters.values()
        ])
        
        distances = torch.norm(cluster_centers - latent_representation, dim=1)
        min_distance, min_idx = torch.min(distances, dim=0)
        nearest_cluster_id = cluster_ids[min_idx.item()]

        # If the nearest cluster is too far, create a new cluster
        if min_distance.item() > self.cluster_distance_threshold:
            new_cluster_id = max(cluster_ids) + 1
            self.optimizer_state.parameter_clusters[new_cluster_id]['members'].append(
                ClusterMember(parameter_vector, performance_score, latent_representation)
            )
            self.optimizer_state.parameter_clusters[new_cluster_id]['center'] = latent_representation
        else:
            # Otherwise, add to the nearest cluster and update its center
            cluster_info = self.optimizer_state.parameter_clusters[nearest_cluster_id]
            cluster_info['members'].append(
                ClusterMember(parameter_vector, performance_score, latent_representation)
            )
            n = len(cluster_info['members'])
            cluster_info['center'] = (cluster_info['center'] * (n - 1) + latent_representation) / n

    # -------------------- Evaluation & Optimization --------------------

    def evaluate_image(self, image_path: Path) -> dict[str, Any]:
        """
        Optimizes OCR parameters for a single image.
        
        This involves:
        - Suggesting initial parameter vectors
        - Evaluating and refining them iteratively
        - Recording the best-found configuration

        Args:
            image_path: Path to the image file to be evaluated.

        Returns:
            dict[str, Any]: Dictionary containing the best parameters, score, and OCR results.
        """
        logger.info(f"Optimizing parameters for {image_path.name}")
        
        # Initialize best_record with very low initial score
        best_record = OptimizationRecord(
            image_path    = image_path,
            parameters    = torch.zeros(len(self.parameter_boundaries)).to(self.device_type),
            score         = -float('inf'),
            latent_vector = torch.zeros(64).to(self.device_type),
            ocr_results   = []
        )

        self.optimizer_state.model.eval()

        def evaluate_parameter_vector(parameter_vector: torch.Tensor) -> OptimizationRecord:
            """
            Evaluates a single parameter vector by applying it to the extractor, running OCR,
            computing a performance score, and encoding the vector in latent space.
            
            Args:
                parameter_vector: The parameter configuration vector to evaluate.
            
            Returns:
                OptimizationRecord capturing performance and OCR results for these parameters.
            """
            params_override = self.vector_to_parameter_dictionary(parameter_vector)
            self.extractor.initialize_processing_steps(config_override = params_override)
            
            # Perform OCR and evaluate performance (longer text, repeated multiple times = higher score)
            ocr_results_raw = self.extractor.perform_ocr_headless([image_path])
            ocr_results     = ocr_results_raw.get(image_path.name, [])
            performance_score = sum(len(text) * count for text, count in ocr_results)
            
            # Encode the parameter vector in latent space
            with torch.inference_mode():
                latent_representation, _ = self.optimizer_state.model(parameter_vector.unsqueeze(0))
                latent_representation    = latent_representation.squeeze()
            
            ocr_results_instances = OCRResult.from_tuples(ocr_results)
            
            return OptimizationRecord(
                image_path    = image_path,
                parameters    = parameter_vector,
                score         = float(performance_score),
                ocr_results   = ocr_results_instances,
                latent_vector = latent_representation 
                    if latent_representation is not None 
                    else torch.zeros(len(self.parameter_boundaries)).to(self.device_type)
            )

        # -------------------- Initial Evaluation --------------------

        for parameter_vector in self.suggest_initial_parameters():
            current_record = evaluate_parameter_vector(parameter_vector)
            best_record.update_if_better(current_record)
            self.update_parameter_clusters(parameter_vector, current_record.score)

        # -------------------- Safety Check --------------------
        
        if best_record.score == -float('inf'):
            logger.warning(
                f"No parameter vectors improved the score for {image_path.name}. "
                f"Using a random parameter vector as a fallback."
            )
            best_parameters = torch.rand(len(self.parameter_boundaries)).to(self.device_type)
            fallback_record = evaluate_parameter_vector(best_parameters)
            fallback_record.score = 0.0
            
            best_record.update_if_better(fallback_record)
            self.update_parameter_clusters(best_parameters, fallback_record.score)

        # -------------------- Iterative Refinement --------------------
        
        for iteration in range(self.iteration_count):
            # Generate candidate parameters by adding noise that decreases over iterations
            candidate_parameter_vectors = [
                torch.clamp(
                    best_record.parameters + torch.randn_like(best_record.parameters) *
                    (1.0 - iteration / self.iteration_count) * 0.1,
                    min=0,
                    max=1
                ) for _ in range(10)
            ]

            candidate_tensor = torch.stack(candidate_parameter_vectors).to(self.device_type)
            means, stds      = self.predict_with_uncertainty(candidate_tensor, num_samples=10)

            # Use UCB to pick the candidate balancing exploitation (mean) and exploration (std)
            ucb_values              = means + self.ucb_beta * stds
            best_candidate_index    = ucb_values.argmax()
            chosen_parameter_vector = candidate_parameter_vectors[best_candidate_index]

            chosen_record = evaluate_parameter_vector(chosen_parameter_vector)

            if best_record.update_if_better(chosen_record):
                logger.info(f"New best score {chosen_record.score:.3f} on iteration {iteration + 1}")

            self.update_parameter_clusters(chosen_parameter_vector, chosen_record.score)

        # -------------------- Record Optimization Outcome --------------------

        self.optimizer_state.optimization_history.append(best_record)
        self.training_buffer.append(best_record)

        # Update global score scaling with the new result
        self.optimizer_state.update_scaling(best_record.score)

        # If training buffer is full, retrain the meta-learner
        if len(self.training_buffer) >= self.training_buffer_size:
            self.train_meta_learner()
            self.training_buffer = []

        # Store final result for quick access
        self.optimization_results[str(image_path)] = best_record.to_dict()
        
        return best_record.to_dict()

    def optimize(self, image_file_paths: list[Path]):
        """
        Optimizes parameters for a batch of images.
        
        Processes each image individually, updates the meta-learning model with the accumulated data,
        saves results and state upon completion.

        Args:
            image_file_paths: List of image file paths to optimize.
        """
        logger.info(f"Beginning meta-learning parameter search for {len(image_file_paths)} images")
        
        for image_path in image_file_paths:
            self.evaluate_image(image_path)
        
        # Train on any remaining buffer and save final state
        if self.training_buffer:
            self.train_meta_learner()
            self.training_buffer = []
        
        self.save_optimization_results()
        self.save_optimizer_state()
        
        logger.info("Meta-learning parameter optimization complete")

    # -------------------- Training --------------------

    def train_meta_learner(self):
        """
        Trains the meta-learning model on historical optimization data.
        
        Key stability features include:
        - Train/validation split using random_split
        - Early stopping to prevent overfitting
        - Gradient clipping for stability
        - Using score scaling to normalize targets
        - Logging progress for monitoring
        """
        history_length = len(self.optimizer_state.optimization_history)
        if history_length < self.training_batch_size:
            logger.info(f"Insufficient data for training. Need at least {self.training_batch_size} samples.")
            return

        # Prepare normalized training data
        parameters = torch.stack([r.parameters for r in self.optimizer_state.optimization_history])
        scores     = torch.tensor([r.score for r in self.optimizer_state.optimization_history], dtype=torch.float32)
        
        min_score, max_score = self.optimizer_state.score_scaling
        if max_score - min_score == 0:
            logger.warning("All scores are the same. Skipping score scaling.")
            scaled_scores = scores
        else:
            scaled_scores = (scores - min_score) / (max_score - min_score)
        
        dataset = TensorDataset(parameters, scaled_scores)

        # Split dataset into training and validation sets (80/20)
        train_size = int(0.8 * history_length)
        valid_size = history_length - train_size
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size = self.training_batch_size,
            shuffle    = True,
            pin_memory = True if self.device_type == 'cuda' else False
        )
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size = self.training_batch_size,
            shuffle    = False,
            pin_memory = True if self.device_type == 'cuda' else False
        )

        # Training parameters
        best_valid_loss  = float('inf')
        patience         = 5
        patience_counter = 0
        max_epochs       = 100
        diversity_weight = 0.1
        grad_clip_norm   = 1.0
        min_improvement  = 1e-4

        self.optimizer_state.model.train()

        for epoch in range(1, max_epochs + 1):
            train_loss = 0.0
            for batch_parameters, batch_scores in train_loader:
                batch_parameters = batch_parameters.to(self.device_type)
                batch_scores     = batch_scores.to(self.device_type)

                self.model_optimizer.zero_grad(set_to_none = True)

                # Forward pass
                latent_representations, predicted_scores = self.optimizer_state.model(batch_parameters)
                
                # Compute MSE loss
                prediction_loss = mse_loss(predicted_scores.squeeze(-1), batch_scores)
                
                # Add diversity loss if multiple samples are present
                if batch_parameters.size(0) > 1:
                    pairwise_distances = torch.pdist(latent_representations)
                    diversity_loss     = -pairwise_distances.mean()
                    total_loss         = prediction_loss + diversity_weight * diversity_loss
                else:
                    total_loss = prediction_loss

                # Backprop with gradient clipping
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.optimizer_state.model.parameters(),
                    max_norm = grad_clip_norm
                )
                self.model_optimizer.step()

                train_loss += total_loss.item()

            average_train_loss = train_loss / len(train_loader)

            # Validation step
            self.optimizer_state.model.eval()
            valid_loss = 0.0
            with torch.inference_mode():
                for batch_parameters, batch_scores in valid_loader:
                    batch_parameters = batch_parameters.to(self.device_type)
                    batch_scores     = batch_scores.to(self.device_type)
                    
                    _, predicted_scores = self.optimizer_state.model(batch_parameters)
                    loss = mse_loss(predicted_scores.squeeze(), batch_scores)
                    valid_loss += loss.item()

            average_valid_loss = valid_loss / len(valid_loader)

            logger.info(
                f"Epoch {epoch}: Train Loss = {average_train_loss:.4f}, Validation Loss = {average_valid_loss:.4f}"
            )

            # Early stopping logic
            if average_valid_loss < best_valid_loss - min_improvement:
                best_valid_loss  = average_valid_loss
                patience_counter = 0
                logger.info(f"Epoch {epoch}: New best validation loss: {best_valid_loss:.4f}")
            else:
                patience_counter += 1
            
            self.learning_rate_scheduler.step(average_valid_loss)
            
            if patience_counter >= patience:
                logger.info(
                    f"Early stopping triggered after {epoch} epochs. "
                    f"Best validation loss: {best_valid_loss:.4f}"
                )
                break

            self.optimizer_state.model.train()

        logger.info(
            f"Training completed. Best validation loss: {best_valid_loss:.4f} "
            f"after {epoch} epochs."
        )

    # -------------------- Uncertainty Estimation --------------------

    @torch.inference_mode()
    def predict_with_uncertainty(
        self, 
        candidate_tensor : torch.Tensor, 
        num_samples      : int = 10
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Estimates predictive uncertainty by performing multiple stochastic forward passes
        (enabling dropout). This helps gauge how confident the model is in its predictions.
        
        Args:
            candidate_tensor : A batch of parameter vectors to evaluate.
            num_samples      : Number of forward passes to approximate uncertainty. Default is 10.

        Returns:
            (means, stds): Tensors containing the mean and standard deviation of predicted scores
                           for each candidate in candidate_tensor.
        """
        self.optimizer_state.model.train()  # Enable dropout for stochasticity
        
        predictions = torch.zeros((num_samples, candidate_tensor.size(0)), device = self.device_type)
        
        # Multiple forward passes to gather a distribution of predictions
        for i in range(num_samples):
            _, predicted_scores = self.optimizer_state.model(candidate_tensor)
            predictions[i] = predicted_scores.squeeze()

        means = predictions.mean(dim=0)
        stds  = predictions.std(dim=0)

        self.optimizer_state.model.eval()
        return means, stds
