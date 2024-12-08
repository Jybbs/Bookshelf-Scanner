# ConfigOptimizer

The **ConfigOptimizer** module introduces a **meta-learning and Bayesian optimization-inspired approach** to identifying optimal OCR processing configurations. By translating each pipeline configuration into a continuous vector representation and employing a learned model to predict OCR performance, we can efficiently navigate the complex parameter space. This stands in contrast to brute-force or exhaustive search methods, which quickly become infeasible as the number of parameters grows.

---

## Theoretical and Mathematical Foundations

Modern OCR pipelines, like **TextExtractor**, contain multiple processing steps—each with a variety of adjustable parameters. Exploring all possible parameter combinations is computationally intractable. Instead, we need a strategy that **learns from experience**, guiding us toward configurations that deliver high-quality OCR outputs without brute-forcing the entire search space.

---

### Defining the Optimization Problem

#### Configuration Space

- Let there be $`S`$ steps in the OCR pipeline.

- Each step $`i`$ has $`P_i`$ parameters.
- Define the total number of parameters as:
  
  $`D = \sum_{i=1}^S P_i.`$
  
We represent any full configuration as a vector of normalized parameters:

$`\mathbf{x} \in [0,1]^D.`$

#### Objective

We aim to find the parameter vector $`\mathbf{x}`$ that maximizes OCR quality:

$`\max_{\mathbf{x} \in [0,1]^D} \text{score}(\mathbf{x}).`$

---

### Scoring OCR Outputs

Given a configuration $`\mathbf{x}`$, the OCR system produces text/confidence pairs:

$`\mathcal{R}(\mathbf{x}) = \{(t_k, c_k)\},`$

where each pair consists of a text snippet $`t_k`$ and a confidence score $`c_k`$.

#### Score Definition

We define the score as:

$`\text{score}(\mathbf{x}) = \sum_{(t,c) \in \mathcal{R}(\mathbf{x})} |t| \cdot c,`$

where $`|t|`$ is the length of the extracted text snippet. This formulation rewards configurations that produce longer, more confidently recognized text, effectively capturing OCR quality in a single scalar value.

---

### Surrogate Modeling for Efficient Search

Instead of running the full OCR process for every candidate configuration, we build a **surrogate model** that predicts scores based on observed data.

#### Surrogate Model Training

- Let $`f_\theta(\mathbf{x})`$ approximate $`\text{score}(\mathbf{x})`$.
- We train $`f_\theta`$ on previously evaluated configurations and their observed scores $`y = \text{score}(\mathbf{x})`$.

The training objective is to minimize the mean squared error:

$`L(\theta) = \mathbb{E}_{(\mathbf{x},y)}[(f_\theta(\mathbf{x}) - y)^2].`$

As we accumulate more training data, our model $`f_\theta`$ guides us toward promising configurations, reducing the need to evaluate unproductive ones.

---

### Infusing Bayesian Optimization Ideas

Classical Bayesian Optimization uses Gaussian Processes to model unknown functions and acquisition functions to select new query points. We capture a similar flavor by using a neural network with dropout to estimate uncertainty.

#### Dropout for Uncertainty Estimates

By applying dropout at inference time, we generate multiple stochastic predictions for the same input $`\mathbf{x}`$:

$`f_\theta^{(k)}(\mathbf{x}), \quad k=1,\dots,K.`$

From these $`K`$ samples, we estimate:

- **Mean Prediction:**
  
  $`\mu(\mathbf{x}) = \frac{1}{K}\sum_{k=1}^K f_\theta^{(k)}(\mathbf{x})`$
  
- **Uncertainty (Standard Deviation):**
  
  $`\sigma(\mathbf{x}) = \sqrt{\frac{1}{K}\sum_{k=1}^K (f_\theta^{(k)}(\mathbf{x}) - \mu(\mathbf{x}))^2}.`$

These estimates provide a "posterior-like" distribution over scores, similar in spirit to Bayesian approaches.

#### Acquisition Function (UCB)

We use an Upper Confidence Bound (UCB) to select new configurations to evaluate:

$`\text{UCB}(\mathbf{x}) = \mu(\mathbf{x}) + \beta \sigma(\mathbf{x}),`$

where $`\beta > 0`$ balances exploration (focusing on configurations with high uncertainty $`\sigma(\mathbf{x})`$) and exploitation (focusing on configurations with a high mean prediction $`\mu(\mathbf{x})`$).

---

### Efficiency in High-Dimensional Spaces

#### Complexity Reduction

If each parameter had $`n`$ possible values, naive brute force would cost $`O(n^D)`$ evaluations. With our surrogate-driven approach:

- We only evaluate $`T \ll n^D`$ configurations.

- Each evaluation refines $`f_\theta`$, improving subsequent guidance.

This yields dramatic savings in both time and computational resources, making the search feasible even in high-dimensional parameter spaces.

---

### Clustering and Latent Representations

Beyond direct parameter optimization, we can analyze the configuration space structurally.

#### Latent Embedding and Clustering

We embed each configuration $`\mathbf{x}`$ into a latent space:

$`\mathbf{x} \xrightarrow{e_\phi} \mathbf{z} \in \mathbb{R}^L.`$

Clustering these latent vectors $`\mathbf{z}`$ helps us:

- Identify clusters of configurations that yield high OCR scores.

- Focus exploration on promising neighborhoods.
- Avoid getting stuck in local optima by providing a higher-level view of the parameter space.

---

## Getting Started

1. **Image Preparation:**  
   Place images (e.g., `0.jpg`, `1.jpg`, etc.) in `images/books`. Supported formats include JPG, PNG, and BMP.

2. **Run the Optimizer:**  
   ```bash
   poetry run config-optimizer
   ```
   The optimizer will automatically propose configurations, run OCR, train the model, and improve suggestions over time.

3. **View Results:**  
   Detailed results are saved in `optimizer.json`, including best configurations and OCR outcomes.

---

## Implementation & Object Details

### Key Components

#### **`ConfigOptimizer` (Main Class):**

Coordinates the entire process:
  
  - Extracts configuration boundaries from `TextExtractor`.

  - Normalizes parameters to $`[0,1]^D`$.
  - Manages the meta-learning model and training buffer.
  - Employs acquisition functions (UCB) and clustering for guided search.
  - Saves and reloads states, ensuring progress persistence.

#### **`MetaLearningModel`:** 

  A neural model composed of:

  - **`ConfigEncoder`:** Maps $`\mathbf{x}`$ to $\mathbf{z}`$, a compressed representation capturing essential features.

  - **`PerformancePredictor`:** Maps $`\mathbf{z}`$ to a predicted score, $`\hat{y}`$.
  
  Together:

  $`f_\theta(\mathbf{x}) = p_\psi(e_\phi(\mathbf{x})).`$

#### **Clustering (`config_clusters`):**  

  Maintains a set of previously evaluated configurations in latent space. Clusters guide the initial suggestions, ensuring both stable exploitation and creative exploration.

#### **`OptimizationRecord`, `ClusterMember`, `OCRResult`:**  

  Data classes used to store evaluations, best configurations, OCR outputs, and cluster information. They maintain a historical record that supports retraining and analysis.

#### **`extract_config_space()` and Normalization:**  

  Extracts parameter ranges from the pipeline and normalizes them. Each parameter originally in $`[a,b]`$ is scaled to $`[0,1]`$:

  $`x_{\text{norm}} = \frac{x - a}{b - a}.`$

#### **`vector_to_config_dictionary()`:**  

  Converts the normalized vector $`\mathbf{x}`$ back into a human-readable dictionary of steps and parameters, ensuring transparency and reproducibility of the final results.

#### **`predict_with_uncertainty()`:**  

  Runs multiple forward passes with dropout to estimate $`\mu(\mathbf{x})$ and $\sigma(\mathbf{x})`$, enabling the UCB acquisition function.

---

## Detailed Output Format

The output `optimizer.json` maps each image to its best configuration and results:
```json
{
  "0.jpg": {
    "config_dict": {
      "steps": {
        "shadow_removal": {
          "enabled": true,
          "parameters": {
            "shadow_kernel_size": { "value": 23 }
          }
        },
        "color_clahe": {
          "enabled": true,
          "parameters": {
            "clahe_clip_limit": { "value": 2.0 }
          }
        }
      }
    },
    "ocr_results": [
      {
        "text": "ANNE FRANK THE DIARY OF A YOUNG GIRL",
        "confidence": 0.7769
      }
    ],
    "score": 26.41
  }
}
```

This final record:
- Clearly shows the chosen configuration in terms of human-understandable steps and parameters.

- Provides the final OCR output and computed score.
- Allows reproducing the exact best configuration for future runs.

---

## Why is This Useful?

- **High-Dimensional Efficiency:** Tames exponentially large search spaces using intelligent, learning-based approaches.

- **Data-Driven Approach:** The more it sees, the better it predicts, reducing guesswork.
- **Bayesian-Inspired Precision:** Balances trying new regions to gain information with refining known good areas.
- **Practical Payoff:** Saves tremendous amounts of computational time and ensures stable, high-quality OCR configurations with fewer trials.
