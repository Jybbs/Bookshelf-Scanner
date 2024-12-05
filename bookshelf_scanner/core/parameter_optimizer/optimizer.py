import json
import torch
import torch.nn as nn

from collections         import defaultdict
from contextlib          import contextmanager
from dataclasses         import dataclass
from pathlib             import Path
from torch.nn.functional import mse_loss
from torch.utils.data    import Dataset, DataLoader
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
    dropout, culminating in a Sigmoid activation to output a normalized score.
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
            nn.Linear(64, 1),
            nn.Sigmoid()
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

# -------------------- Dataset --------------------


class OptimizationHistoryDataset(Dataset):
    """
    Dataset of historical parameter configurations and their OCR scores.

    Each record contains:
        - 'parameters' : Parameter vector (torch.Tensor).
        - 'score'      : Observed OCR performance score (float).

    This dataset is used to train the MetaLearningModel.
    """
    def __init__(self, records: list[dict[str, Any]]):
        self.parameter_vectors  = [
            torch.tensor(record['parameters'], dtype=torch.float32) 
            for record in records
        ]
        self.performance_scores = [
            torch.tensor(record['score'], dtype=torch.float32) 
            for record in records
        ]

    def __len__(self) -> int:
        return len(self.parameter_vectors)
        
    def __getitem__(self, index: int):
        return self.parameter_vectors[index], self.performance_scores[index]

# -------------------- Data Classes --------------------


@dataclass
class OptimizationRecord:
    """
    Records the outcome of a single optimization attempt for an image.

    Attributes:
        image_path    : Path to the image being optimized.
        parameters    : Best-found parameter vector (torch.Tensor).
        score         : OCR performance score achieved (float).
        latent_vector : Latent representation of the parameters (torch.Tensor).
    """
    image_path    : Path
    parameters    : torch.Tensor
    score         : float
    latent_vector : torch.Tensor

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
    optimization_history : list[OptimizationRecord]                                  = None
    parameter_clusters   : dict[int, list[tuple[torch.Tensor, float, torch.Tensor]]] = None
    score_scaling        : tuple[float, float]                                       = (0.0, 1.0)

    def __post_init__(self):
        if self.optimization_history is None:
            self.optimization_history = []
        if self.parameter_clusters is None:
            self.parameter_clusters = defaultdict(list)
        
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


class MetaLearningOptimizer:
    """
    Coordinates meta-learning guided parameter optimization for OCR.

    This optimizer leverages historical optimization data to suggest and refine
    parameter configurations that maximize OCR performance. It manages the meta-learning
    model, clusters parameter configurations, and handles training and persistence.

    Attributes:
        extractor_instance      : Instance of TextExtractor being optimized.
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
    MODEL_DIRECTORY        = PROJECT_ROOT_DIRECTORY / 'bookshelf_scanner' / 'models'
    OUTPUT_JSON_FILE       = RESULTS_DIRECTORY / 'optimizer.json'
    MODEL_PYTORCH_FILE     = MODEL_DIRECTORY / 'meta_learner.pt'
    
    # -------------------- Initialization & State --------------------
    
    def __init__(
        self,
        extractor_instance    : TextExtractor,
        device_type           : str   = 'cuda' if torch.cuda.is_available() else 'cpu',
        initial_points_count  : int   = 10,
        iteration_count       : int   = 40,
        training_batch_size   : int   = 32,
        learning_rate_value   : float = 1e-3
    ):
        """
        Initializes the MetaLearningOptimizer.

        Args:
            extractor_instance    : TextExtractor instance to optimize.
            device_type           : Device for computations ('cpu' or 'cuda').
            initial_points_count  : Number of initial parameter suggestions.
            iteration_count       : Number of refinement steps per image.
            training_batch_size   : Batch size for training.
            learning_rate_value   : Learning rate for the optimizer.
        """
        self.extractor_instance      = extractor_instance
        self.device_type             = device_type
        self.initial_points_count    = initial_points_count
        self.iteration_count         = iteration_count
        self.training_batch_size     = training_batch_size
        self.learning_rate_value     = learning_rate_value
        self.parameter_boundaries    = self.extract_parameter_boundaries()
        
        input_dimension               = len(self.parameter_boundaries)
        self.optimizer_state          = MetaLearningState(
            model=self.initialize_meta_learning_model(input_dimension)
        )
        
        # Initialize optimizer and scheduler
        self.model_optimizer          = torch.optim.Adam(
            self.optimizer_state.model.parameters(), 
            lr=self.learning_rate_value
        )
        self.learning_rate_scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.model_optimizer, 
            factor=0.5, 
            patience=5, 
            verbose=False
        )
        
        self.optimization_results     = {}
        
        # Load existing state if available
        self.load_optimizer_state()
    
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
        processing_steps      = self.extractor_instance.initialize_processing_steps()
        parameter_boundaries  = []
        
        for processing_step in processing_steps:
            if processing_step.parameters and processing_step.is_pipeline:
                for parameter in processing_step.parameters:
                    parameter_boundaries.append({
                        'name'        : f"{processing_step.name}.{parameter.name}",
                        'min_value'   : float(parameter.min),
                        'max_value'   : float(parameter.max),
                        'step_value'  : float(parameter.step),
                        'is_integer'  : isinstance(parameter.value, int)
                    })

        return parameter_boundaries
    
    def vector_to_parameter_dictionary(self, parameter_vector: torch.Tensor) -> dict[str, dict[str, Any]]:
        """
        Converts a normalized parameter vector to a parameter dictionary.

        Scales each value from [0,1] to its original range and rounds if necessary.

        Args:
            parameter_vector: Normalized parameter vector (torch.Tensor).

        Returns:
            Dictionary structured for the extractor's parameter override.
        """
        parameter_dictionary = {}
        vector_numpy         = parameter_vector.cpu().numpy()

        for index, boundary in enumerate(self.parameter_boundaries):
            step_name, parameter_name = boundary['name'].split('.')
            
            # Initialize step in dictionary if not present
            if step_name not in parameter_dictionary:
                parameter_dictionary[step_name] = {
                    'enabled'   : True,
                    'parameters': {}
                }

            # Scale the parameter value
            scaled_value = (
                vector_numpy[index] * 
                (boundary['max_value'] - boundary['min_value']) + 
                boundary['min_value']
            )
            scaled_value = int(round(scaled_value)) if boundary['is_integer'] else round(scaled_value, 1)
            
            # Assign the scaled value to the parameter dictionary
            parameter_dictionary[step_name]['parameters'][parameter_name] = scaled_value

        return parameter_dictionary
    
    # -------------------- Persistence & State Management --------------------
    
    def load_optimizer_state(self):
        """
        Loads the meta-learning state from disk if available.

        Restores the model's weights, optimization history, and parameter clusters.
        """
        if self.MODEL_PYTORCH_FILE.exists():
            checkpoint = torch.load(self.MODEL_PYTORCH_FILE, map_location=self.device_type)
            
            # Load model state
            self.optimizer_state.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Restore optimization history
            self.optimizer_state.optimization_history = [
                OptimizationRecord(
                    image_path    = Path(record['image_path']),
                    parameters    = torch.tensor(record['parameters'], dtype=torch.float32),
                    score         = record['score'],
                    latent_vector = torch.tensor(record['latent_vector'], dtype=torch.float32)
                ) for record in checkpoint['optimization_history']
            ]

            # Restore parameter clusters
            parameter_clusters = defaultdict(list)
            for cluster_id, members in checkpoint['parameter_clusters'].items():
                cluster_id_int = int(cluster_id)
                for member in members:
                    parameter_clusters[cluster_id_int].append((
                        torch.tensor(member['parameters'], dtype=torch.float32),
                        member['score'],
                        torch.tensor(member['latent'], dtype=torch.float32)
                    ))

            self.optimizer_state.parameter_clusters = parameter_clusters
            self.optimizer_state.score_scaling      = tuple(checkpoint['score_scaling'])
            
            logger.info(f"State loaded from {self.MODEL_PYTORCH_FILE}")
    
    def save_optimizer_state(self):
        """
        Saves the current meta-learning state and model to disk.

        Includes model weights, optimization history, parameter clusters, and score scaling.
        """
        self.MODEL_PYTORCH_FILE.parent.mkdir(exist_ok=True, parents=True)
        
        # Prepare data for saving
        checkpoint_data = {
            'model_state_dict'     : self.optimizer_state.model.state_dict(),

            'optimization_history' : [
                {
                    'image_path'    : str(record.image_path),
                    'parameters'    : record.parameters.tolist(),
                    'score'         : record.score,
                    'latent_vector' : record.latent_vector.tolist()
                } for record in self.optimizer_state.optimization_history
            ],

            'parameter_clusters'   : {
                cluster_id: [
                    {
                        'parameters' : parameter.tolist(),
                        'score'      : score,
                        'latent'     : latent.tolist()
                    } for parameter, score, latent in members
                ] for cluster_id, members in self.optimizer_state.parameter_clusters.items()
            },

            'score_scaling'        : self.optimizer_state.score_scaling
        }
        
        # Save the checkpoint
        torch.save(checkpoint_data, self.MODEL_PYTORCH_FILE)
        logger.info(f"Model and state saved to {self.MODEL_PYTORCH_FILE}")
    
    def save_optimization_results(self):
        """
        Saves the optimization results to a JSON file.

        Records the best parameters and scores for each processed image.
        """
        output_data = {
            Path(image_path).name: {
                'parameters' : optimization_record['parameters'],
                'score'      : optimization_record['score']
            }
            for image_path, optimization_record in self.optimization_results.items()
        }
        
        # Ensure the output directory exists
        self.OUTPUT_JSON_FILE.parent.mkdir(exist_ok=True, parents=True)
        
        # Write the results to a JSON file
        with self.OUTPUT_JSON_FILE.open('w') as output_file:
            json.dump(output_data, output_file, indent=2)
        
        logger.info(f"Results saved to {self.OUTPUT_JSON_FILE}")
    
    # -------------------- Parameter Clustering & Suggestion --------------------
    
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
        for cluster_members in self.optimizer_state.parameter_clusters.values():
            if cluster_members:
                best_member = max(cluster_members, key=lambda member: member[1])[0]
                suggested_parameters.append(best_member)

        # If there are existing suggestions, add slight variations
        if suggested_parameters:
            best_parameter_vector = suggested_parameters[0]
            
            while len(suggested_parameters) < self.initial_points_count:
                noise                = torch.randn_like(best_parameter_vector) * 0.1
                new_parameter_vector = torch.clamp(best_parameter_vector + noise, 0, 1)
                suggested_parameters.append(new_parameter_vector)
        else:
            # If no existing suggestions, generate random parameter vectors
            for _ in range(self.initial_points_count):
                suggested_parameters.append(torch.rand(len(self.parameter_boundaries)))

        return suggested_parameters
    
    def update_parameter_clusters(self, parameter_vector: torch.Tensor, performance_score: float):
        """
        Assigns parameter configurations to latent clusters.

        Encodes parameters to latent space and assigns them to the nearest cluster or
        creates a new cluster if no close match exists.

        Args:
            parameter_vector : Parameter vector (torch.Tensor).
            performance_score : OCR performance score (float).
        """
        self.optimizer_state.model.eval()
        
        # Encode the parameter vector to latent space
        with torch.no_grad():
            latent_representation, _ = self.optimizer_state.model(parameter_vector.unsqueeze(0))

        latent_representation = latent_representation.squeeze().cpu()
        
        # If no clusters exist, create the first cluster
        if not self.optimizer_state.parameter_clusters:
            self.optimizer_state.parameter_clusters[0] = [
                (parameter_vector, performance_score, latent_representation)
            ]
            return
        
        # Find the nearest cluster based on latent representation
        minimum_distance   = float('inf')
        nearest_cluster_id = None
        
        for cluster_id, cluster_members in self.optimizer_state.parameter_clusters.items():
            cluster_center = torch.stack([member[2] for member in cluster_members]).mean(0)
            distance       = torch.norm(latent_representation - cluster_center)
            
            if distance < minimum_distance:
                minimum_distance  = distance
                nearest_cluster_id = cluster_id

        # Define a threshold to decide whether to create a new cluster
        if minimum_distance > 2.0:
            new_cluster_id = (
                max(self.optimizer_state.parameter_clusters.keys()) + 1 
                if self.optimizer_state.parameter_clusters 
                else 0
            )
            self.optimizer_state.parameter_clusters[new_cluster_id] = [
                (parameter_vector, performance_score, latent_representation)
            ]

        else:
            self.optimizer_state.parameter_clusters[nearest_cluster_id].append(
                (parameter_vector, performance_score, latent_representation)
            )
    
    # -------------------- Optimization & Training --------------------
    
    def evaluate_image(self, image_path: Path) -> dict[str, Any]:
        """
        Optimizes OCR parameters for a single image.

        Performs initial evaluations and iterative refinements to find the best
        parameter configuration that maximizes OCR performance.

        Args:
            image_path: Path to the image to optimize.

        Returns:
            Dictionary containing the best parameters and achieved score.
        """
        logger.info(f"Optimizing parameters for {image_path.name}")
        best_score           = 0.0
        best_parameters      = None
        best_latent_vector   = None

        # -------------------- Initial Evaluation --------------------
        
        for parameter_vector in self.suggest_initial_parameters():
            # Convert vector to parameter dictionary
            parameter_dictionary = self.vector_to_parameter_dictionary(parameter_vector)
            
            # Initialize processing steps with the suggested parameters
            self.extractor_instance.initialize_processing_steps(
                params_override=parameter_dictionary
            )
            
            # Perform OCR and calculate performance score
            ocr_results       = self.extractor_instance.perform_ocr_headless([image_path])
            performance_score = sum(
                len(text) * count 
                for text, count in ocr_results.get(image_path.name, [])
            )

            # Update best score and parameters if current score is better
            if performance_score > best_score:
                best_score         = performance_score
                best_parameters    = parameter_vector
                
                with torch.no_grad():
                    best_latent_vector, _ = self.optimizer_state.model(parameter_vector.unsqueeze(0))
            
            # Update parameter clusters with the current evaluation
            self.update_parameter_clusters(parameter_vector, performance_score)

        # -------------------- Iterative Refinement --------------------
        
        for iteration in range(self.iteration_count):
            candidate_parameter_vectors = []
            
            # Generate candidate parameter vectors with decreasing noise
            for _ in range(10):
                noise                = torch.randn_like(best_parameters) * (1.0 - iteration / self.iteration_count) * 0.1
                candidate_vector     = torch.clamp(best_parameters + noise, 0, 1)
                candidate_parameter_vectors.append(candidate_vector)
            
            self.optimizer_state.model.eval()
            
            # Predict performance scores for candidate parameters
            with torch.no_grad():
                candidate_tensor = torch.stack(candidate_parameter_vectors).to(self.device_type)
                _, predictions   = self.optimizer_state.model(candidate_tensor)
            
            # Select the best candidate based on predicted scores
            best_candidate_index    = predictions.squeeze().argmax()
            chosen_parameter_vector = candidate_parameter_vectors[best_candidate_index]
            
            # Convert chosen vector to parameter dictionary
            chosen_parameter_dictionary = self.vector_to_parameter_dictionary(chosen_parameter_vector)
            
            # Initialize processing steps with the chosen parameters
            self.extractor_instance.initialize_processing_steps(
                params_override=chosen_parameter_dictionary
            )
            
            # Perform OCR and calculate performance score
            ocr_results       = self.extractor_instance.perform_ocr_headless([image_path])
            performance_score = sum(
                len(text) * count 
                for text, count in ocr_results.get(image_path.name, [])
            )
            
            # Update best score and parameters if current score is better
            if performance_score > best_score:
                best_score          = performance_score
                best_parameters     = chosen_parameter_vector
                
                with torch.no_grad():
                    best_latent_vector, _ = self.optimizer_state.model(chosen_parameter_vector.unsqueeze(0))
                
                logger.info(f"New best score {performance_score:.3f} on iteration {iteration + 1}")
            
            # Update parameter clusters with the current evaluation
            self.update_parameter_clusters(chosen_parameter_vector, performance_score)

        # -------------------- Record Optimization Outcome --------------------
        
        self.optimizer_state.optimization_history.append(
            OptimizationRecord(
                image_path    = image_path,
                parameters    = best_parameters,
                score         = best_score,
                latent_vector = (
                    best_latent_vector.squeeze() 
                    if best_latent_vector is not None 
                    else torch.zeros(len(self.parameter_boundaries))
                )
            )
        )
        
        # Update score scaling based on the new score
        self.optimizer_state.update_scaling(best_score)
        
        # Train the meta-learning model with updated history
        self.train_meta_learner()

        # Store the final parameters and score in results
        final_parameter_dictionary = self.vector_to_parameter_dictionary(best_parameters)
        self.optimization_results[str(image_path)] = {
            'parameters' : final_parameter_dictionary,
            'score'      : best_score
        }
        
        return {
            'parameters' : final_parameter_dictionary,
            'score'      : best_score
        }
    
    def optimize(self, image_file_paths: list[Path]):
        """
        Optimizes parameters for a batch of images.

        Processes each image, updates the meta-learning model, and saves results and state.

        Args:
            image_file_paths: List of image file paths to optimize.
        """
        logger.info(f"Beginning meta-learning parameter search for {len(image_file_paths)} images")
        
        # Use context manager to handle graceful exit
        with self.graceful_exit():
            for image_path in image_file_paths:
                self.evaluate_image(image_path)
            
            # Save all optimization results and current state
            self.save_optimization_results()
            self.save_optimizer_state()
        
        logger.info("Meta-learning parameter optimization complete")
    
    def train_meta_learner(self):
        """
        Trains the meta-learning model on historical optimization data.

        Uses past parameter configurations and their scores to refine the model,
        enhancing its ability to predict performance and guide future optimizations.
        """
        # Check if there is enough data to train
        if len(self.optimizer_state.optimization_history) < self.training_batch_size:
            return
        
        # Create dataset from optimization history
        dataset = OptimizationHistoryDataset([
            {
                'parameters' : record.parameters.tolist(),
                'score'      : record.score
            } for record in self.optimizer_state.optimization_history
        ])
        
        # Initialize data loader
        data_loader = DataLoader(dataset, batch_size=self.training_batch_size, shuffle=True)
        self.optimizer_state.model.train()
        total_loss = 0.0
        
        # Iterate over batches
        for batch_parameters, batch_scores in data_loader:
            batch_parameters = batch_parameters.to(self.device_type)
            batch_scores     = batch_scores.to(self.device_type)
            
            # Zero gradients
            self.model_optimizer.zero_grad()
            
            # Forward pass
            latent_representations, predicted_scores = self.optimizer_state.model(batch_parameters)
            
            # Compute loss
            loss = mse_loss(predicted_scores.squeeze(), batch_scores)
            
            # Encourage latent diversity if batch has multiple samples
            if batch_parameters.size(0) > 1:
                latent_pairwise_distances = torch.pdist(latent_representations)
                diversity_loss            = -latent_pairwise_distances.mean()
                loss += 0.1 * diversity_loss
            
            # Backward pass and optimization step
            loss.backward()
            self.model_optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
        
        # Calculate average loss and update scheduler
        average_loss = total_loss / len(data_loader)
        self.learning_rate_scheduler.step(average_loss)
        
        logger.info(f"Meta-learner training loss: {average_loss:.4f}")
    
    # -------------------- User Input Handling --------------------
    
    @contextmanager
    def graceful_exit(self):
        """
        Ensures graceful shutdown by saving state upon interruption.

        If a KeyboardInterrupt or exception occurs, the current state and results
        are saved before re-raising the exception.
        """
        try:
            yield

        except KeyboardInterrupt:
            logger.info("\nOptimization interrupted. Saving current results and state...")
            self.save_optimization_results()
            self.save_optimizer_state()
            raise

        except Exception as exception:
            logger.error(f"Optimization failed: {exception}")
            self.save_optimization_results()
            self.save_optimizer_state()
            raise
