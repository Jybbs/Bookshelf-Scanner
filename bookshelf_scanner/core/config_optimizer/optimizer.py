import json
import numpy    as np
import torch
import torch.nn as nn

from bookshelf_scanner   import ModuleLogger, TextExtractor, Utils
from collections         import defaultdict
from dataclasses         import dataclass
from omegaconf           import OmegaConf
from pathlib             import Path
from torch.nn.functional import mse_loss
from torch.utils.data    import DataLoader, random_split, TensorDataset
from typing              import Any

logger = ModuleLogger('optimizer')()

# -------------------- Neural Network Components --------------------

class ConfigEncoder(nn.Module):
    """
    Encodes full configuration vectors (representing steps and their parameters) into latent representations.

    This encoder reduces the dimensionality of a configuration vector that encodes multiple steps
    and their associated parameters. It captures essential features that influence OCR performance.
    The architecture consists of two linear layers with ReLU activations and dropout for regularization.
    """
    def __init__(
        self, 
        input_dimension    : int, 
        latent_dimension   : int = 64, 
        encoder_hidden_dim : int = 128
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, encoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(encoder_hidden_dim, latent_dimension),
            nn.ReLU()
        )

    def forward(self, config_vector: torch.Tensor) -> torch.Tensor:
        return self.network(config_vector)

class PerformancePredictor(nn.Module):
    """
    Predicts OCR performance from latent configuration representations.

    This predictor estimates the OCR performance score based on latent embeddings produced
    by the ConfigEncoder. It uses a series of linear layers with ReLU activations and dropout,
    culminating in an output layer without activation to allow unrestricted score ranges.
    """
    def __init__(
        self, 
        latent_dimension       : int = 64, 
        predictor_hidden_dim_1 : int = 128, 
        predictor_hidden_dim_2 : int = 64
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dimension, predictor_hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(predictor_hidden_dim_1, predictor_hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(predictor_hidden_dim_2, 1)
        )
        
    def forward(self, latent_representation: torch.Tensor) -> torch.Tensor:
        return self.network(latent_representation)

class MetaLearningModel(nn.Module):
    """
    Combines ConfigEncoder and PerformancePredictor for meta-learning.

    This model encodes configuration vectors (combinations of steps and parameters)
    into latent vectors and predicts their corresponding OCR performance scores.
    The modular design allows for separate training and fine-tuning of the encoder and predictor.
    """
    def __init__(
        self,
        input_dimension        : int,
        latent_dimension       : int = 64,
        encoder_hidden_dim     : int = 128,
        predictor_hidden_dim_1 : int = 128,
        predictor_hidden_dim_2 : int = 64
    ):
        super().__init__()
        self.config_encoder        = ConfigEncoder(input_dimension, latent_dimension, encoder_hidden_dim)
        self.performance_predictor = PerformancePredictor(latent_dimension, predictor_hidden_dim_1, predictor_hidden_dim_2)
        
    def forward(self, config_vector: torch.Tensor):
        latent_representation = self.config_encoder(config_vector)
        performance_score     = self.performance_predictor(latent_representation)
        return latent_representation, performance_score

# -------------------- Data Classes --------------------

@dataclass
class ClusterMember:
    """
    Represents a member within a configuration cluster.
    
    Attributes:
        config_vector     : The configuration vector representing steps and parameters.
        performance_score : The OCR performance score achieved with this configuration.
        latent_vector     : The latent representation of the configuration vector.
    """
    config_vector     : torch.Tensor
    performance_score : float
    latent_vector     : torch.Tensor

    @classmethod
    def from_dict(cls, data: dict, device_type: str = 'cpu') -> 'ClusterMember':
        """
        Creates a ClusterMember instance from a dictionary.

        Args:
            data: Dictionary containing 'parameters' (the config vector), 'score', and 'latent' keys.

        Returns:
            ClusterMember instance.
        """
        return cls(
            config_vector     = torch.tensor(data['parameters'], dtype = torch.float32, device = device_type),
            performance_score = data['score'],
            latent_vector     = torch.tensor(data['latent'], dtype = torch.float32, device = device_type)
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
    def from_dict(cls, data: dict, device_type: str = 'cpu') -> 'OCRResult':
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
        config_vector : Best-found configuration vector (torch.Tensor).
        score         : OCR performance score achieved (float).
        latent_vector : Latent representation of the configuration (torch.Tensor).
        ocr_results   : List of OCRResult instances containing extracted text and confidence.
    """
    image_path    : Path
    config_vector : torch.Tensor
    score         : float
    latent_vector : torch.Tensor
    ocr_results   : list[OCRResult]

    def to_dict(self) -> dict:
        """
        Converts the OptimizationRecord instance into a dictionary suitable for JSON serialization.
        """
        return {
            'image_path'    : str(self.image_path),
            'parameters'    : self.config_vector.tolist(),
            'score'         : self.score,
            'latent_vector' : self.latent_vector.tolist(),
            'ocr_results'   : [ocr.to_dict() for ocr in self.ocr_results]
        }
    
    @classmethod
    def from_dict(cls, data: dict, device_type: str = 'cpu') -> 'OptimizationRecord':
        """
        Creates an OptimizationRecord instance from a dictionary.

        Args:
            data : A dictionary containing 'image_path', 'parameters', 'score', 'latent_vector', and 'ocr_results'.

        Returns:
            OptimizationRecord: An instance populated with the provided data.
        """
        return cls(
            image_path    = Path(data['image_path']),
            config_vector = torch.tensor(data['parameters'], dtype = torch.float32, device = device_type),
            score         = data['score'],
            latent_vector = torch.tensor(data['latent_vector'], dtype = torch.float32, device = device_type),
            ocr_results   = [OCRResult.from_dict(ocr, device_type = device_type) for ocr in data.get('ocr_results', [])]
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
            self.config_vector = other.config_vector
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
        config_clusters      : Clusters of similar configuration latent vectors.
        score_scaling        : Tuple tracking (min, max) scores encountered.
    """
    model                : MetaLearningModel
    optimization_history : list[OptimizationRecord]  = None
    config_clusters      : dict[int, dict[str, Any]] = None
    score_scaling        : tuple[float, float]        = (0.0, 1.0)

    def __post_init__(self):
        if self.optimization_history is None:
            self.optimization_history = []

        if self.config_clusters is None:
            self.config_clusters = defaultdict(lambda: {'members': [], 'center': None})
        
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

class ConfigOptimizer:
    """
    Coordinates meta-learning guided configuration optimization for OCR.
    
    This optimizer leverages historical optimization data to suggest and refine
    configurations that maximize OCR performance. It manages the meta-learning model,
    clusters configurations, and handles training and persistence.

    All parameters are now loaded from a YAML file using OmegaConf.
    """

    # -------------------- Project & File Directories --------------------
    PROJECT_ROOT_DIRECTORY = Utils.find_root('pyproject.toml')
    RESULTS_DIRECTORY      = PROJECT_ROOT_DIRECTORY / 'bookshelf_scanner' / 'data' / 'results'
    MODEL_DIRECTORY        = PROJECT_ROOT_DIRECTORY / 'bookshelf_scanner' / 'core' / 'config_optimizer' / 'models'
    OUTPUT_JSON_FILE       = RESULTS_DIRECTORY / 'optimizer.json'
    PARAMS_FILE            = PROJECT_ROOT_DIRECTORY / 'bookshelf_scanner' / 'config' / 'optimizer.yml'
    MODEL_PYTORCH_FILE     = MODEL_DIRECTORY / 'meta_learner.pt'

    def __init__(
        self, 
        extractor     : TextExtractor, 
        output_images : bool = False, 
        output_json   : bool = True
    ):
        """
        Initializes the ConfigOptimizer, loading all configuration parameters from optimizer.yml.

        Args:
            extractor     : TextExtractor instance to optimize (required).
            output_images : Whether to output the best processed image after optimization.
            output_json   : Whether to output results to a JSON file.
        """
        # Load configuration from YAML file
        config = OmegaConf.load(self.PARAMS_FILE)

        # Optimization parameters
        self.cluster_distance_threshold     = config.optimization.cluster_distance_threshold
        self.device_type                    = config.optimization.device_type
        self.extractor                      = extractor
        self.initial_points_count           = config.optimization.initial_points_count
        self.initial_suggestion_noise_scale = config.optimization.initial_suggestion_noise_scale
        self.iteration_count                = config.optimization.iteration_count
        self.learning_rate_value            = config.optimization.learning_rate_value
        self.refinement_candidate_count     = config.optimization.refinement_candidate_count
        self.refinement_noise_scale         = config.optimization.refinement_noise_scale
        self.training_batch_size            = config.optimization.training_batch_size
        self.training_buffer_size           = config.optimization.training_buffer_size
        self.ucb_beta                       = config.optimization.ucb_beta

        # Training parameters
        self.diversity_weight               = config.training.diversity_weight
        self.early_stopping_patience        = config.training.early_stopping_patience
        self.grad_clip_norm                 = config.training.grad_clip_norm
        self.lr_scheduler_factor            = config.training.lr_scheduler_factor
        self.lr_scheduler_patience          = config.training.lr_scheduler_patience
        self.max_epochs                     = config.training.max_epochs
        self.min_improvement                = config.training.min_improvement
        self.train_valid_split_ratio        = config.training.train_valid_split_ratio

        # Architecture parameters
        self.encoder_hidden_dim             = config.architecture.encoder_hidden_dim
        self.latent_dimension               = config.architecture.latent_dimension
        self.predictor_hidden_dim_1         = config.architecture.predictor_hidden_dim_1
        self.predictor_hidden_dim_2         = config.architecture.predictor_hidden_dim_2

        # Uncertainty parameters
        self.uncertainty_num_samples        = config.uncertainty.uncertainty_num_samples

        # Output toggles
        self.output_images = output_images
        self.output_json   = output_json

        # Directly use extractor.config_space, which is pre-computed at extractor initialization
        self.config_space_boundaries = self.extractor.config_space
        input_dimension              = len(self.config_space_boundaries)
        self.optimizer_state         = MetaLearningState(model = self.initialize_meta_learning_model(input_dimension))
        
        # Initialize the meta-learner optimizer and scheduler
        self.model_optimizer = torch.optim.Adam(
            self.optimizer_state.model.parameters(), 
            lr = self.learning_rate_value
        )
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.model_optimizer, 
            factor   = self.lr_scheduler_factor, 
            patience = self.lr_scheduler_patience
        )
        
        self.optimization_results = {}
        
        # Load existing state if available
        self.load_optimizer_state()
        
        # Initialize training buffer for batching training samples
        self.training_buffer = []

    def initialize_meta_learning_model(self, input_dimension: int) -> MetaLearningModel:
        """
        Initializes the MetaLearningModel with the given input dimension and architecture parameters.

        Args:
            input_dimension: The dimension of the configuration vector.

        Returns:
            An instance of MetaLearningModel.
        """
        return MetaLearningModel(
            input_dimension,
            latent_dimension       = self.latent_dimension,
            encoder_hidden_dim     = self.encoder_hidden_dim,
            predictor_hidden_dim_1 = self.predictor_hidden_dim_1,
            predictor_hidden_dim_2 = self.predictor_hidden_dim_2
        ).to(device = self.device_type)

    # -------------------- Configuration Space & Conversion --------------------

    def extract_config_space(self, config_state) -> list[dict[str, Any]]:
        """
        Extracts configuration range definitions from the given ConfigState.
        
        The ConfigState includes multiple steps, each with several parameters.
        We flatten those into a list of parameter definitions so we can map each parameter
        to a continuous or discrete range and represent the entire configuration as a vector.

        Args:
            config_state: A ConfigState instance from which to extract parameter definitions.

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
            for step_name, step_definition in config_state.config_dict["steps"].items()
            if step_definition.get("parameters") is not None
            for param_name, param_definition in step_definition["parameters"].items()
        ]
    
    def vector_to_config_dictionary(self, config_vector: torch.Tensor) -> dict[str, Any]:
        """
        Converts a normalized configuration vector back into a human-readable configuration dictionary.

        Each entry in the config vector corresponds to a parameter of a particular step.
        We scale each normalized value [0,1] back to its original range and round if necessary.
        
        Args:
            config_vector: Normalized configuration vector (torch.Tensor).

        Returns:
            Dictionary structured for the extractor's config override, containing steps and parameters.
        """
        config_dict = {"steps": {}}
        vector_numpy = config_vector.detach().cpu().numpy()  # Numpy vectors require a CPU conversion

        names      = [b['name']       for b in self.config_space_boundaries]
        min_values = np.array([b['min_value']  for b in self.config_space_boundaries])
        max_values = np.array([b['max_value']  for b in self.config_space_boundaries])
        is_integer = np.array([b['is_integer'] for b in self.config_space_boundaries])

        scaled_values = vector_numpy * (max_values - min_values) + min_values
        scaled_values = np.where(is_integer, np.round(scaled_values).astype(int), np.round(scaled_values, 1))

        for name, val in zip(names, scaled_values):
            step_name, param_name = name.split('.')
            if step_name not in config_dict["steps"]:
                config_dict["steps"][step_name] = {
                    'enabled'    : True,
                    'parameters' : {}
                }
            config_dict["steps"][step_name]['parameters'][param_name] = {
                "value": int(val) if isinstance(val, np.integer) else float(val)
            }

        return config_dict
    
    # -------------------- Persistence & State Management --------------------

    def load_optimizer_state(self):
        """
        Loads the meta-learning state from disk if available.
        
        This restores the model's weights, optimization history (including OCR results),
        configuration clusters, and score scaling. After loading, it recalculates cluster centers
        to ensure no cluster center is None.
        """
        if self.MODEL_PYTORCH_FILE.exists():
            checkpoint = torch.load(self.MODEL_PYTORCH_FILE, map_location = self.device_type)

            # Load model state
            self.optimizer_state.model.load_state_dict(checkpoint['model_state_dict'])

            # Restore optimization history
            self.optimizer_state.optimization_history = [
                OptimizationRecord.from_dict(record, device_type = self.device_type)
                for record in checkpoint['optimization_history']
            ]

            # Restore configuration clusters
            config_clusters = defaultdict(lambda: {'members': [], 'center': None})
            for cluster_id_str, members in checkpoint['parameter_clusters'].items():
                cluster_id_int = int(cluster_id_str)
                cluster_members = [ClusterMember.from_dict(m, device_type = self.device_type) for m in members]
                config_clusters[cluster_id_int]['members'].extend(cluster_members)

            # Recalculate cluster centers now that we have members
            for _, cinfo in config_clusters.items():
                if cinfo['members']:
                    latent_vectors = torch.stack([m.latent_vector for m in cinfo['members']])
                    cinfo['center'] = latent_vectors.mean(dim = 0).to(device = self.device_type)
                else:
                    cinfo['center'] = None

            self.optimizer_state.config_clusters = config_clusters

            # Restore score scaling
            self.optimizer_state.score_scaling = tuple(checkpoint['score_scaling'])
            
            logger.info(f"State loaded from {self.MODEL_PYTORCH_FILE}")

    def save_optimizer_state(self):
        """
        Saves the current meta-learning state and model to disk.
        
        Includes model weights, optimization history (with OCR results), configuration clusters, and score scaling.
        """
        self.MODEL_PYTORCH_FILE.parent.mkdir(exist_ok = True, parents = True)
        
        checkpoint_data = {
            'model_state_dict'     : self.optimizer_state.model.state_dict(),
            'optimization_history' : [record.to_dict() for record in self.optimizer_state.optimization_history],
            'parameter_clusters'   : {
                cluster_id: [
                    {
                        'parameters' : member.config_vector.tolist(),
                        'score'      : member.performance_score,
                        'latent'     : member.latent_vector.tolist()
                    } for member in cluster_info['members']
                ] for cluster_id, cluster_info in self.optimizer_state.config_clusters.items()
            },
            'score_scaling' : self.optimizer_state.score_scaling
        }
        
        torch.save(checkpoint_data, self.MODEL_PYTORCH_FILE)
        logger.info(f"Model and state saved to {self.MODEL_PYTORCH_FILE}")

    def save_optimization_results(self):
        """
        Saves a simplified version of the optimization results to a JSON file.

        For each image, we store:
        - The image name (e.g. "0.jpg") as the key in the JSON.
        - A 'config_dict' with human-readable steps and parameters.
        - The 'score' achieved.
        - The 'ocr_results' list.

        This excludes internal vectors like the latent representation to produce a more
        human-friendly and reproducible configuration record.
        """
        if not self.output_json:
            return

        output_data = {}
        for record in self.optimizer_state.optimization_history:
            image_key = record.image_path.name

            # Convert best config vector back to a human-readable config dictionary
            config_dict = self.vector_to_config_dictionary(record.config_vector)

            output_data[image_key] = {
                'config_dict' : config_dict,
                'ocr_results' : [ocr.to_dict() for ocr in record.ocr_results],
                'score'       : record.score
            }

        self.OUTPUT_JSON_FILE.parent.mkdir(exist_ok = True, parents = True)
        with self.OUTPUT_JSON_FILE.open('w', encoding = 'utf-8') as output_file:
            json.dump(output_data, output_file, indent = 2, ensure_ascii = False)

        logger.info(f"Results saved to {self.OUTPUT_JSON_FILE}")

    # -------------------- Configuration Suggestion & Clustering --------------------

    @torch.inference_mode()
    def suggest_initial_configurations(self) -> list[torch.Tensor]:
        """
        Suggests initial configurations for evaluation.
        
        Combines best configurations from existing clusters with random samples to balance
        exploration (new random points) and exploitation (best cluster configurations).
        
        Returns:
            List of configuration vectors (torch.Tensor) to evaluate.
        """
        suggested_configs = []

        # Add best configurations from existing clusters
        for cluster_info in self.optimizer_state.config_clusters.values():
            members = cluster_info['members']
            if members:
                best_member = max(members, key = lambda member: member.performance_score)
                suggested_configs.append(best_member.config_vector)

        # If existing suggestions are available, diversify around the best found configuration
        if suggested_configs:
            best_config_vector = suggested_configs[0]
            needed = self.initial_points_count - len(suggested_configs)
            for _ in range(needed):
                noise = torch.randn_like(best_config_vector, device = self.device_type) * self.initial_suggestion_noise_scale
                suggested_configs.append(torch.clamp(best_config_vector + noise, 0, 1))
        else:
            # If no existing suggestions, generate purely random configuration vectors
            suggested_configs.extend(
                torch.rand(len(self.config_space_boundaries), device = self.device_type) for _ in range(self.initial_points_count)
            )

        return suggested_configs

    @torch.inference_mode()
    def update_config_clusters(
        self, 
        config_vector     : torch.Tensor, 
        performance_score : float
    ):
        """
        Assigns configuration vectors to latent clusters in the learned latent space.
        
        Encodes the configuration vector to a latent space, then assigns it to the nearest cluster or
        creates a new cluster if no close match exists.
        
        Args:
            config_vector     : Configuration vector.
            performance_score : OCR performance score achieved with this configuration.
        """
        self.optimizer_state.model.eval()

        # Encode the configuration vector into latent space for clustering
        latent_representation, _ = self.optimizer_state.model(config_vector.unsqueeze(0))
        latent_representation = latent_representation.squeeze().to(device = self.device_type)

        # If no clusters exist, initialize the first cluster
        if not self.optimizer_state.config_clusters:
            self.optimizer_state.config_clusters[0]['members'].append(
                ClusterMember(config_vector, performance_score, latent_representation)
            )
            self.optimizer_state.config_clusters[0]['center'] = latent_representation
            return
        
        # Compute distances to existing cluster centers
        cluster_ids     = list(self.optimizer_state.config_clusters.keys())
        cluster_centers = torch.stack([
            cluster_info['center'] 
            for cluster_info in self.optimizer_state.config_clusters.values()
        ])
        
        distances = torch.norm(cluster_centers - latent_representation, dim = 1)
        min_distance, min_idx = torch.min(distances, dim = 0)
        nearest_cluster_id = cluster_ids[min_idx.item()]

        # If the nearest cluster is too far, create a new cluster
        if min_distance.item() > self.cluster_distance_threshold:
            new_cluster_id = max(cluster_ids) + 1
            self.optimizer_state.config_clusters[new_cluster_id]['members'].append(
                ClusterMember(config_vector, performance_score, latent_representation)
            )
            self.optimizer_state.config_clusters[new_cluster_id]['center'] = latent_representation
        else:
            # Otherwise, add to the nearest cluster and update its center
            cluster_info = self.optimizer_state.config_clusters[nearest_cluster_id]
            cluster_info['members'].append(
                ClusterMember(config_vector, performance_score, latent_representation)
            )
            n = len(cluster_info['members'])
            cluster_info['center'] = (cluster_info['center'] * (n - 1) + latent_representation) / n

    # -------------------- Evaluation & Optimization --------------------

    def evaluate_image(self, image_path: Path) -> dict[str, Any]:
        """
        Optimizes OCR configurations for a single image.
        
        This involves:
        - Suggesting initial configuration vectors
        - Evaluating and refining them iteratively
        - Recording the best-found configuration

        Args:
            image_path: Path to the image file to be evaluated.

        Returns:
            dict[str, Any]: Dictionary containing the best config, score, and OCR results.
        """
        logger.info(f"Optimizing configuration for {image_path.name}")
        
        # Initialize best_record with very low initial score
        best_record = OptimizationRecord(
            image_path    = image_path,
            config_vector = torch.zeros(len(self.config_space_boundaries), device = self.device_type),
            score         = -float('inf'),
            latent_vector = torch.zeros(self.latent_dimension, device = self.device_type),
            ocr_results   = []
        )

        self.optimizer_state.model.eval()

        def evaluate_config_vector(config_vector: torch.Tensor) -> OptimizationRecord:
            """
            Evaluates a single configuration vector by applying it to the extractor, running OCR,
            computing a performance score, and encoding the vector in latent space.
            
            Args:
                config_vector: The configuration vector to evaluate.
            
            Returns:
                OptimizationRecord capturing performance and OCR results for this configuration.
            """
            config_vector = config_vector.to(device = self.device_type)
            config_override = self.vector_to_config_dictionary(config_vector)
            
            # Directly run headless mode on the single image and get results
            
            original_output_images = self.extractor.output_images
            self.extractor.output_images = False # output_images is off during search to avoid saving intermediate images
            results = self.extractor.run_headless_mode([image_path], config_override = config_override)
            self.extractor.output_images = original_output_images

            # Extract OCR results for the current image
            image_name        = image_path.name
            ocr_tuples        = results.get(image_name, [])
            performance_score = sum(len(text) * confidence for text, confidence in ocr_tuples)
            ocr_results_instances = OCRResult.from_tuples(ocr_tuples)

            # Encode the config vector in latent space
            with torch.inference_mode():
                latent_representation, _ = self.optimizer_state.model(config_vector.unsqueeze(0))
                latent_representation    = latent_representation.squeeze()
            
            return OptimizationRecord(
                image_path    = image_path,
                config_vector = config_vector,
                score         = float(performance_score),
                ocr_results   = ocr_results_instances,
                latent_vector = latent_representation 
                    if latent_representation is not None 
                    else torch.zeros(len(self.config_space_boundaries), device = self.device_type)
            )

        # -------------------- Initial Evaluation --------------------

        for config_vector in self.suggest_initial_configurations():
            current_record = evaluate_config_vector(config_vector)
            best_record.update_if_better(current_record)
            self.update_config_clusters(config_vector, current_record.score)

        # -------------------- Safety Check --------------------
        
        if best_record.score == -float('inf'):
            logger.warning(
                f"No configurations improved the score for {image_path.name}. "
                f"Using a random configuration vector as a fallback."
            )
            random_config         = torch.rand(len(self.config_space_boundaries), device = self.device_type)
            fallback_record       = evaluate_config_vector(random_config)
            fallback_record.score = 0.0
            
            best_record.update_if_better(fallback_record)
            self.update_config_clusters(random_config, fallback_record.score)

        # -------------------- Iterative Refinement --------------------
        
        for iteration in range(self.iteration_count):
            # Generate candidate configurations by adding noise that decreases over iterations
            candidate_config_vectors = [
                torch.clamp(
                    best_record.config_vector + torch.randn_like(best_record.config_vector, device = self.device_type) *
                    (1.0 - iteration / self.iteration_count) * self.refinement_noise_scale,
                    min = 0,
                    max = 1
                ) for _ in range(self.refinement_candidate_count)
            ]

            candidate_tensor = torch.stack(candidate_config_vectors).to(device = self.device_type)
            means, stds      = self.predict_with_uncertainty(candidate_tensor, num_samples = self.uncertainty_num_samples)

            # Use UCB to pick the candidate balancing exploitation (mean) and exploration (std)
            ucb_values           = means + self.ucb_beta * stds
            best_candidate_index = ucb_values.argmax()
            chosen_config_vector = candidate_config_vectors[best_candidate_index]

            chosen_record = evaluate_config_vector(chosen_config_vector)

            if best_record.update_if_better(chosen_record):
                logger.info(f"New best score {chosen_record.score:.3f} on iteration {iteration + 1}")

            self.update_config_clusters(chosen_config_vector, chosen_record.score)

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

        # -------------------- Save Best Image if Requested --------------------

        if self.output_images:
            config_override                 = self.vector_to_config_dictionary(best_record.config_vector)
            original_output_images          = self.extractor.output_images
            original_output_dir             = self.extractor.output_image_dir
            self.extractor.output_images    = True
            self.extractor.run_headless_mode([image_path], config_override = config_override)
            self.extractor.output_images    = original_output_images
            self.extractor.output_image_dir = original_output_dir
        
        return best_record.to_dict()

    def optimize(self, image_file_paths: list[Path]):
        """
        Optimizes configurations for a batch of images.
        
        Processes each image individually, updates the meta-learning model with the accumulated data,
        saves results and state upon completion.

        Args:
            image_file_paths: List of image file paths to optimize.
        """
        logger.info(f"Beginning meta-learning configuration search for {len(image_file_paths)} images")
        
        for image_path in image_file_paths:
            self.evaluate_image(image_path)
        
        # Train on any remaining buffer and save final state
        if self.training_buffer:
            self.train_meta_learner()
            self.training_buffer = []
        
        self.save_optimization_results()
        self.save_optimizer_state()
        
        logger.info("Meta-learning configuration optimization complete")

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
        config_vectors = torch.stack([r.config_vector for r in self.optimizer_state.optimization_history]).to(device = self.device_type)
        scores         = torch.tensor([r.score for r in self.optimizer_state.optimization_history], dtype = torch.float32, device = self.device_type)
        
        min_score, max_score = self.optimizer_state.score_scaling
        if max_score - min_score == 0:
            logger.warning("All scores are the same. Skipping score scaling.")
            scaled_scores = scores
        else:
            scaled_scores = (scores - min_score) / (max_score - min_score)
        
        dataset = TensorDataset(config_vectors, scaled_scores)

        # Split dataset into training and validation sets
        train_size = int(self.train_valid_split_ratio * history_length)
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

        best_valid_loss  = float('inf')
        patience_counter = 0

        self.optimizer_state.model.train()

        for epoch in range(1, self.max_epochs + 1):
            train_loss = 0.0
            for batch_configs, batch_scores in train_loader:
                batch_configs = batch_configs.to(device = self.device_type)
                batch_scores  = batch_scores.to(device = self.device_type)

                self.model_optimizer.zero_grad(set_to_none = True)

                # Forward pass
                latent_representations, predicted_scores = self.optimizer_state.model(batch_configs)
                
                # Compute MSE loss
                prediction_loss = mse_loss(predicted_scores.squeeze(-1), batch_scores)
                
                # Add diversity loss if multiple samples are present
                if batch_configs.size(0) > 1:
                    pairwise_distances = torch.pdist(latent_representations)
                    diversity_loss     = -pairwise_distances.mean()
                    total_loss         = prediction_loss + self.diversity_weight * diversity_loss
                else:
                    total_loss = prediction_loss

                # Backprop with gradient clipping
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.optimizer_state.model.parameters(),
                    max_norm = self.grad_clip_norm
                )
                self.model_optimizer.step()

                train_loss += total_loss.item()

            average_train_loss = train_loss / len(train_loader)

            # Validation step
            self.optimizer_state.model.eval()
            valid_loss = 0.0
            with torch.inference_mode():
                for batch_configs, batch_scores in valid_loader:
                    batch_configs = batch_configs.to(device = self.device_type)
                    batch_scores  = batch_scores.to(device = self.device_type)
                    
                    _, predicted_scores = self.optimizer_state.model(batch_configs)
                    loss = mse_loss(predicted_scores.squeeze(), batch_scores)
                    valid_loss += loss.item()

            average_valid_loss = valid_loss / len(valid_loader)

            logger.info(
                f"Epoch {epoch}: Train Loss = {average_train_loss:.4f}, Validation Loss = {average_valid_loss:.4f}"
            )

            # Early stopping logic
            if average_valid_loss < best_valid_loss - self.min_improvement:
                best_valid_loss  = average_valid_loss
                patience_counter = 0
                logger.info(f"Epoch {epoch}: New best validation loss: {best_valid_loss:.4f}")
            else:
                patience_counter += 1
            
            self.learning_rate_scheduler.step(average_valid_loss)
            
            if patience_counter >= self.early_stopping_patience:
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
        num_samples      : int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Estimates predictive uncertainty by performing multiple stochastic forward passes
        (enabling dropout). This helps gauge how confident the model is in its predictions.
        
        Args:
            candidate_tensor : A batch of configuration vectors to evaluate.
            num_samples      : Number of forward passes to approximate uncertainty.

        Returns:
            (means, stds): Tensors containing the mean and standard deviation of predicted scores
                           for each candidate in candidate_tensor.
        """
        self.optimizer_state.model.train()  # Enable dropout for stochastic passes
        
        predictions = torch.zeros((num_samples, candidate_tensor.size(0)), device = self.device_type)
        
        # Multiple forward passes to gather a distribution of predictions
        for i in range(num_samples):
            _, predicted_scores = self.optimizer_state.model(candidate_tensor)
            predictions[i] = predicted_scores.squeeze()

        means = predictions.mean(dim = 0)
        stds  = predictions.std(dim = 0)

        self.optimizer_state.model.eval()
        return means, stds
