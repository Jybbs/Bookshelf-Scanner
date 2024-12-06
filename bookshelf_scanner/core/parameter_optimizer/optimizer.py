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
        ocr_results   : List of tuples containing (text, confidence score).
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
            data : A dictionary containing 'image_path', 'parameters', 'score', 'latent_vector', and 'ocr_results' keys.

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

# -------------------- Meta-Learning Optimizer Class --------------------

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
    
    # -------------------- Class Constants --------------------
    
    PROJECT_ROOT_DIRECTORY = Utils.find_root('pyproject.toml')
    RESULTS_DIRECTORY      = PROJECT_ROOT_DIRECTORY / 'bookshelf_scanner' / 'data' / 'results'
    MODEL_DIRECTORY        = PROJECT_ROOT_DIRECTORY / 'bookshelf_scanner' / 'parameter_optimizer' / 'models'
    OUTPUT_JSON_FILE       = RESULTS_DIRECTORY / 'optimizer.json'
    MODEL_PYTORCH_FILE     = MODEL_DIRECTORY / 'meta_learner.pt'
    
    # -------------------- Initialization & State --------------------
    
    def __init__(
        self,
        extractor                  : TextExtractor,
        device_type                : str   = 'cuda' if torch.cuda.is_available() else 'cpu',
        initial_points_count       : int   = 10,
        iteration_count            : int   = 40,
        training_batch_size        : int   = 16,
        learning_rate_value        : float = 1e-3,
        ucb_beta                   : float = 0.1,
        cluster_distance_threshold : float = 2.0,
        training_buffer_size       : int   = 30,
    ):
        """
        Initializes the MetaLearningOptimizer.

        Args:
            extractor                  : TextExtractor instance to optimize.
            device_type                : Device for computations ('cpu' or 'cuda').
            initial_points_count       : Number of initial parameter suggestions.
            iteration_count            : Number of refinement steps per image.
            training_batch_size        : Batch size for training.
            learning_rate_value        : Learning rate for the optimizer.
            ucb_beta                   : Exploration parameter for Upper Confidence Bound (UCB) acquisition.
            cluster_distance_threshold : Distance threshold for clustering.
            training_buffer_size       : Number of samples to accumulate before training the model.
        """
        self.extractor                  = extractor
        self.device_type                = device_type
        self.initial_points_count       = initial_points_count
        self.iteration_count            = iteration_count
        self.training_batch_size        = training_batch_size
        self.learning_rate_value        = learning_rate_value
        self.ucb_beta                   = ucb_beta
        self.cluster_distance_threshold = cluster_distance_threshold
        self.training_buffer_size       = training_buffer_size

        self.parameter_boundaries = self.extract_parameter_boundaries()
        
        input_dimension      = len(self.parameter_boundaries)
        self.optimizer_state = MetaLearningState(
            model = self.initialize_meta_learning_model(input_dimension)
        )
        
        # Initialize optimizer and scheduler
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
        
        # Initialize training buffer to accumulate OptimizationRecords
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
            List of dictionaries defining each parameter's bounds and type.
        """
        processing_steps     = self.extractor.initialize_processing_steps()
        parameter_boundaries = []
        
        for processing_step in processing_steps:
            if processing_step.parameters and processing_step.is_pipeline:
                for parameter in processing_step.parameters:
                    parameter_boundaries.append({
                        'name'       : f"{processing_step.name}.{parameter.name}",
                        'min_value'  : float(parameter.min),
                        'max_value'  : float(parameter.max),
                        'step_value' : float(parameter.step),
                        'is_integer' : isinstance(parameter.value, int)
                    })

        return parameter_boundaries
    
    def vector_to_parameter_dictionary(self, parameter_vector: torch.Tensor) -> dict[str, dict[str, Any]]:
        """
        Converts a normalized parameter vector to a parameter dictionary.

        Scales each value from [0,1] to its original range and rounds if necessary.

        Args:
            parameter_vector: Normalized parameter vector (torch.Tensor).

        Returns:
            Dictionary structured for the extractor's config override.
        """
        parameter_dictionary = {}
        vector_numpy = parameter_vector.cpu().numpy()

        names = [boundary['name'] for boundary in self.parameter_boundaries]
        min_values = np.array([boundary['min_value']  for boundary in self.parameter_boundaries])
        max_values = np.array([boundary['max_value']  for boundary in self.parameter_boundaries])
        is_integer = np.array([boundary['is_integer'] for boundary in self.parameter_boundaries])

        scaled_values = vector_numpy * (max_values - min_values) + min_values
        scaled_values = np.where(is_integer, np.round(scaled_values).astype(int), np.round(scaled_values, 1))

        for name, value in zip(names, scaled_values):
            step_name, parameter_name = name.split('.')
            if step_name not in parameter_dictionary:
                parameter_dictionary[step_name] = {
                    'enabled'    : True,
                    'parameters' : {}
                }
            parameter_dictionary[step_name]['parameters'][parameter_name] = value

        return parameter_dictionary
    
    # -------------------- Persistence & State Management --------------------
    
    def load_optimizer_state(self):
        """
        Loads the meta-learning state from disk if available.

        Restores the model's weights, optimization history (including OCR results), parameter clusters, and score scaling.
        """
        if self.MODEL_PYTORCH_FILE.exists():
            checkpoint = torch.load(self.MODEL_PYTORCH_FILE, map_location = self.device_type)

            # Load model state
            self.optimizer_state.model.load_state_dict(checkpoint['model_state_dict'])

            # Restore optimization history
            self.optimizer_state.optimization_history = [
                OptimizationRecord.from_dict(record)
                for record in checkpoint['optimization_history']
            ]

            # Restore parameter clusters
            parameter_clusters = defaultdict(lambda: {'members': [], 'center': None})
            for cluster_id, members in checkpoint['parameter_clusters'].items():
                cluster_id_int = int(cluster_id)

                for member in members:
                    cluster_member = ClusterMember.from_dict(member)
                    parameter_clusters[cluster_id_int]['members'].append(cluster_member)

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
        
        # Prepare data for saving
        checkpoint_data = {
            'model_state_dict' : self.optimizer_state.model.state_dict(),

            'optimization_history' : [
                record.to_dict() for record in self.optimizer_state.optimization_history
            ],

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
        
        # Save the checkpoint
        torch.save(checkpoint_data, self.MODEL_PYTORCH_FILE)
        logger.info(f"Model and state saved to {self.MODEL_PYTORCH_FILE}")

    def save_optimization_results(self):
        """
        Saves the optimization results to a JSON file.

        Records the best parameters, scores, and OCR results for each processed image.
        """
        output_data = {
            str(record.image_path): record.to_dict()
            for record in self.optimizer_state.optimization_history
        }
        
        self.OUTPUT_JSON_FILE.parent.mkdir(exist_ok = True, parents = True)
        
        with self.OUTPUT_JSON_FILE.open('w', encoding = 'utf-8') as output_file:
            json.dump(output_data, output_file, indent = 2, ensure_ascii = False)
        
        logger.info(f"Results saved to {self.OUTPUT_JSON_FILE}")
   
    # -------------------- Parameter Clustering & Suggestion --------------------
    
    @torch.inference_mode()
    def suggest_initial_parameters(self) -> list[torch.Tensor]:
        """
        Suggests initial parameter configurations for evaluation.

        Combines best parameters from existing clusters with random samples to balance
        exploration and exploitation.

        Returns:
            List of parameter vectors (torch.Tensor) to evaluate.
        """
        suggested_parameters = []

        # Extract best parameters from existing clusters
        for cluster_info in self.optimizer_state.parameter_clusters.values():
            members = cluster_info['members']
            if members:
                # Select the member with the highest performance_score
                best_member = max(members, key=lambda member: member.performance_score)
                suggested_parameters.append(best_member.parameter_vector)

        # If there are existing suggestions, add slight variations
        if suggested_parameters:
            best_parameter_vector = suggested_parameters[0]

            while len(suggested_parameters) < self.initial_points_count:
                noise = torch.randn_like(best_parameter_vector) * 0.1
                new_parameter_vector = torch.clamp(best_parameter_vector + noise, 0, 1)
                suggested_parameters.append(new_parameter_vector)
        else:
            # If no existing suggestions, generate random parameter vectors
            for _ in range(self.initial_points_count):
                suggested_parameters.append(torch.rand(len(self.parameter_boundaries)))

        return suggested_parameters

    @torch.inference_mode()
    def update_parameter_clusters(self, 
        parameter_vector  : torch.Tensor, 
        performance_score : float
    ):
        """
        Assigns parameter configurations to latent clusters.

        Encodes parameters to latent space and assigns them to the nearest cluster or
        creates a new cluster if no close match exists.

        Args:
            parameter_vector  : Parameter configuration vector.
            performance_score : OCR performance score achieved with this configuration.
        """
        self.optimizer_state.model.eval()
        
        # Encode the parameter vector to latent space
        latent_representation, _ = self.optimizer_state.model(parameter_vector.unsqueeze(0))
        latent_representation = latent_representation.squeeze().cpu()  # Shape: (latent_dim,)
        
        # If no clusters exist, create the first cluster
        if not self.optimizer_state.parameter_clusters:
            self.optimizer_state.parameter_clusters[0]['members'].append(
                ClusterMember(parameter_vector, performance_score, latent_representation)
            )
            self.optimizer_state.parameter_clusters[0]['center'] = latent_representation
            return
        
        # Collect existing cluster centers
        cluster_ids     = list(self.optimizer_state.parameter_clusters.keys())
        cluster_centers = torch.stack([
            cluster_info['center'] 
            for cluster_info in self.optimizer_state.parameter_clusters.values()
        ])
        
        # Vectorized distance computation between new latent vector and all cluster centers
        distances = torch.norm(cluster_centers - latent_representation, dim=1)  # Shape: (num_clusters,)
        
        # Find the nearest cluster
        min_distance, min_idx = torch.min(distances, dim=0)
        nearest_cluster_id = cluster_ids[min_idx.item()]
        
        # Use a threshold to decide whether to create a new cluster
        if min_distance.item() > self.cluster_distance_threshold:
            new_cluster_id = max(cluster_ids) + 1
            self.optimizer_state.parameter_clusters[new_cluster_id]['members'].append(
                ClusterMember(parameter_vector, performance_score, latent_representation)
            )
            self.optimizer_state.parameter_clusters[new_cluster_id]['center'] = latent_representation

        else:
            # Append to the nearest cluster and update the center incrementally
            cluster_info = self.optimizer_state.parameter_clusters[nearest_cluster_id]
            cluster_info['members'].append(
                ClusterMember(parameter_vector, performance_score, latent_representation)
            )
            # Update the running average for the cluster center
            n = len(cluster_info['members'])
            cluster_info['center'] = (cluster_info['center'] * (n - 1) + latent_representation) / n
    
    # -------------------- Optimization & Training --------------------
    
    def evaluate_image(self, image_path: Path) -> dict[str, Any]:
        """
        Optimizes OCR parameters for a single image.

        Args:
            image_path : Path to the image file to evaluate.

        Returns:
            dict[str, Any]: A dictionary containing the best parameters, score, and OCR results.
        """
        logger.info(f"Optimizing parameters for {image_path.name}")
        
        # Initialize best_record with default values
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
            Evaluates a single parameter vector by performing OCR and calculating the performance score.
            """
            # Convert vector to parameter dictionary
            self.extractor.initialize_processing_steps(
                params_override = self.vector_to_parameter_dictionary(parameter_vector)
            )
            
            # Perform OCR and calculate performance score
            ocr_results_raw   = self.extractor.perform_ocr_headless([image_path])
            ocr_results       = ocr_results_raw.get(image_path.name, [])
            performance_score = sum(len(text) * count for text, count in ocr_results)
            
            # Encode the parameter vector
            with torch.inference_mode():
                latent_representation, _ = self.optimizer_state.model(parameter_vector.unsqueeze(0))
                latent_representation    = latent_representation.squeeze()
            
            # Convert raw OCR results to OCRResult instances
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
            # Fallback: Use a random parameter vector
            best_parameters = torch.rand(len(self.parameter_boundaries)).to(self.device_type)
            fallback_record = evaluate_parameter_vector(best_parameters)
            fallback_record.score = 0.0
            
            best_record.update_if_better(fallback_record)
            self.update_parameter_clusters(best_parameters, fallback_record.score)

        # -------------------- Iterative Refinement --------------------

        for iteration in range(self.iteration_count):
            # Generate candidate parameter vectors with decreasing noise
            candidate_parameter_vectors = [
                torch.clamp(
                    input = best_record.parameters + torch.randn_like(best_record.parameters) * 
                            (1.0 - iteration / self.iteration_count) * 0.1, 
                    min   = 0, 
                    max   = 1
                )
                for _ in range(10)
            ]

            # Stack candidates into a tensor and predict means/stds
            candidate_tensor = torch.stack(candidate_parameter_vectors).to(self.device_type)
            means, stds      = self.predict_with_uncertainty(candidate_tensor, num_samples=10)

            # Compute UCB acquisition values and select the best candidate
            ucb_values              = means + self.ucb_beta * stds
            best_candidate_index    = ucb_values.argmax()
            chosen_parameter_vector = candidate_parameter_vectors[best_candidate_index]

            chosen_record = evaluate_parameter_vector(chosen_parameter_vector)

            if best_record.update_if_better(chosen_record):
                logger.info(f"New best score {chosen_record.score:.3f} on iteration {iteration + 1}")

            # Update parameter clusters with the current evaluation
            self.update_parameter_clusters(chosen_parameter_vector, chosen_record.score)

        # -------------------- Record Optimization Outcome --------------------

        self.optimizer_state.optimization_history.append(best_record)
        self.training_buffer.append(best_record)  # Add to training buffer

        # Update scoring
        self.optimizer_state.update_scaling(best_record.score)

        # Check if training buffer is full
        if len(self.training_buffer) >= self.training_buffer_size:
            self.train_meta_learner()
            self.training_buffer = []  # Reset training buffer

        # Store the final optimization results
        self.optimization_results[str(image_path)] = best_record.to_dict()
        
        return best_record.to_dict()

    def optimize(self, image_file_paths: list[Path]):
        """
        Optimizes parameters for a batch of images.

        Processes each image, updates the meta-learning model, and saves results and state.

        Args:
            image_file_paths: List of image file paths to optimize.
        """
        logger.info(f"Beginning meta-learning parameter search for {len(image_file_paths)} images")
        
        # Remove graceful exit handling
        for image_path in image_file_paths:
            self.evaluate_image(image_path)
        
        # After processing all images, ensure any remaining data in the training buffer is used for training
        if self.training_buffer:
            self.train_meta_learner()
            self.training_buffer = []  # Reset training buffer
        
        # Save all optimization results and current state
        self.save_optimization_results()
        self.save_optimizer_state()
        
        logger.info("Meta-learning parameter optimization complete")
    
    def train_meta_learner(self):
        """
        Trains the meta-learning model on historical optimization data with essential stability features.
        
        Features:
        1. Train/validation split using torch.utils.data.random_split.
        2. Early stopping to prevent overfitting.
        3. Gradient clipping to manage exploding gradients.
        4. Efficient gradient zeroing with set_to_none=True.
        5. Basic logging for monitoring.
        6. Score Scaling: Normalizes performance scores based on observed min and max.
        """
        # Check if there is enough data to train
        history_length = len(self.optimizer_state.optimization_history)
        if history_length < self.training_batch_size:
            logger.info(f"Insufficient data for training. Need at least {self.training_batch_size} samples.")
            return

        # Prepare training data
        parameters = torch.stack([r.parameters for r in self.optimizer_state.optimization_history])
        scores     = torch.tensor([r.score     for r in self.optimizer_state.optimization_history], dtype = torch.float32)
        
        # Implement score scaling
        min_score, max_score = self.optimizer_state.score_scaling
        if max_score - min_score == 0:
            logger.warning("All scores are the same. Skipping score scaling.")
            scaled_scores = scores
        else:
            scaled_scores = (scores - min_score) / (max_score - min_score)
        
        dataset    = TensorDataset(parameters, scaled_scores)

        # Split data into training and validation sets (80/20 split)
        train_size = int(0.8 * history_length)
        valid_size = history_length - train_size
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

        # Create data loaders
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

        # Initialize training state
        best_valid_loss  = float('inf')
        patience         = 5  # Number of epochs to wait for improvement
        patience_counter = 0
        max_epochs       = 100
        
        # Training constants
        diversity_weight = 0.1
        grad_clip_norm   = 1.0
        min_improvement  = 1e-4  # Minimum improvement to reset patience

        self.optimizer_state.model.train()

        for epoch in range(1, max_epochs + 1):
            # Training phase
            train_loss = 0.0
            for batch_parameters, batch_scores in train_loader:
                batch_parameters = batch_parameters.to(self.device_type)
                batch_scores     = batch_scores.to(self.device_type)

                self.model_optimizer.zero_grad(set_to_none = True)

                # Forward pass
                latent_representations, predicted_scores = self.optimizer_state.model(batch_parameters)
                
                # Compute MSE loss
                prediction_loss = mse_loss(predicted_scores.squeeze(-1), batch_scores)
                
                # Add diversity loss if batch has multiple samples
                if batch_parameters.size(0) > 1:
                    pairwise_distances = torch.pdist(latent_representations)
                    diversity_loss     = -pairwise_distances.mean()  # Negative to maximize distances
                    total_loss         = prediction_loss + diversity_weight * diversity_loss
                else:
                    total_loss = prediction_loss

                # Backward pass with gradient clipping
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.optimizer_state.model.parameters(),
                    max_norm = grad_clip_norm
                )
                self.model_optimizer.step()

                train_loss += total_loss.item()

            average_train_loss = train_loss / len(train_loader)

            # Validation phase
            self.optimizer_state.model.eval()
            valid_loss = 0.0
            with torch.inference_mode():
                for batch_parameters, batch_scores in valid_loader:
                    batch_parameters = batch_parameters.to(self.device_type)
                    batch_scores     = batch_scores.to(self.device_type)
                    
                    # Forward pass
                    _, predicted_scores = self.optimizer_state.model(batch_parameters)
                    loss        = mse_loss(predicted_scores.squeeze(), batch_scores)
                    valid_loss += loss.item()

            average_valid_loss = valid_loss / len(valid_loader)

            logger.info(
                f"Epoch {epoch}: Train Loss = {average_train_loss:.4f}, Validation Loss = {average_valid_loss:.4f}"
            )

            # Early stopping check
            if average_valid_loss < best_valid_loss - min_improvement:
                best_valid_loss  = average_valid_loss
                patience_counter = 0
                logger.info(f"Epoch {epoch}: New best validation loss: {best_valid_loss:.4f}")
            else:
                patience_counter += 1
            
            # Update learning rate scheduler
            self.learning_rate_scheduler.step(average_valid_loss)
            
            # Check early stopping condition
            if patience_counter >= patience:
                logger.info(
                    f"Early stopping triggered after {epoch} epochs. "
                    f"Best validation loss: {best_valid_loss:.4f}"
                )
                break

            self.optimizer_state.model.train()

        # Final training summary
        logger.info(
            f"Training completed. Best validation loss: {best_valid_loss:.4f} "
            f"after {epoch} epochs."
        )
    
    @torch.inference_mode()
    def predict_with_uncertainty(self, 
        candidate_tensor : torch.Tensor, 
        num_samples      : int = 10
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs multiple stochastic forward passes (with dropout enabled) to estimate
        mean and standard deviation of predictions. This simulates uncertainty estimation.

        Args:
            candidate_tensor : A batch of parameter vectors to evaluate.
            num_samples      : Number of forward passes to approximate uncertainty. Default is 10.

        Returns:
            Two torch.Tensors containing the mean and standard deviation of the predicted scores 
            for each candidate in candidate_tensor.
        """
        # Set the model to train mode to enable dropout for stochastic passes
        self.optimizer_state.model.train()
        
        # Preallocate tensor for predictions
        predictions = torch.zeros((num_samples, candidate_tensor.size(0)), device = self.device_type)
        
        # Perform multiple forward passes to gather predictions
        for i in range(num_samples):
            _, predicted_scores = self.optimizer_state.model(candidate_tensor)
            predictions[i] = predicted_scores.squeeze()

        # Compute mean and standard deviation
        means = predictions.mean(dim = 0)
        stds  = predictions.std(dim = 0)

        # Switch back to evaluation mode after uncertainty estimation
        self.optimizer_state.model.eval()

        return means, stds
