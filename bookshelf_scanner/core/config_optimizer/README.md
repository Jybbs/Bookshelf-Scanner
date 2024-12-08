# ConfigOptimizer

The **ConfigOptimizer** module introduces a **meta-learning and Bayesian optimization-inspired approach** to identifying optimal OCR processing configurations. By translating each pipeline configuration into a continuous vector representation and employing a learned model to predict OCR performance, we can efficiently navigate the complex parameter space. This stands in contrast to brute-force or exhaustive search methods, which quickly become infeasible as the number of parameters grows.

---

## Theoretical and Mathematical Foundations

Modern OCR pipelines, like `TextExtractor`, involve multiple processing steps each with numerous adjustable parameters. Naively searching over all combinations is prohibitively expensive and impractical. Instead, we want a method that **learns** as it goes, zeroing in on high-quality configurations without enumerating the entire space.

### Problem Setting

Suppose the pipeline has $`S`$ steps, and step $`i`$ contains $`P_i`$ parameters. Letting  
$`D = \sum_{i=1}^S P_i,`$  
we represent a full configuration as a vector  
$`\mathbf{x} \in [0,1]^D,`$  
where each parameter is normalized to the unit interval for consistency.

Our goal is to find  
$`\max_{\mathbf{x} \in [0,1]^D} \text{score}(\mathbf{x}),`$

where $`\text{score}(\mathbf{x})`$ measures OCR quality under configuration $`\mathbf{x}`$.

### Scoring the OCR Output

Given a configuration $`\mathbf{x}`$, the OCR system produces a set of text/confidence pairs:
$`\mathcal{R}(\mathbf{x}) = \{(t_k, c_k)\}.`$
We define:
$`\text{score}(\mathbf{x}) = \sum_{(t,c) \in \mathcal{R}(\mathbf{x})} |t| \cdot c,`$
where $`|t|`$ is the length of the extracted text $`t`$ and $`c`$ is the confidence score. This rewards configurations that yield both longer text and higher confidence, correlating closely with OCR effectiveness.

### Meta-Learning Surrogate Model

To avoid brute-force search, we train a model to approximate the performance function:
$`f_\theta(\mathbf{x}) \approx \text{score}(\mathbf{x}).`$

This is done by minimizing a mean squared error loss:
$`L(\theta) = \mathbb{E}_{(\mathbf{x},y)}[(f_\theta(\mathbf{x}) - y)^2],`$
where $`y = \text{score}(\mathbf{x})`$ is obtained from actual OCR runs. As we gather more data, $`f_\theta`$ becomes a reliable guide to the configuration space.

### Bayesian Optimization Inspiration

In classical **Bayesian Optimization**, a Gaussian Process (GP) models the unknown objective, and we use an acquisition function to decide where to sample next. Here, we mimic the Bayesian flavor using a neural network and dropout-based uncertainty:

1. **Approximating a Posterior:**
   By enabling dropout at inference, each forward pass of $f_\theta(\mathbf{x})$ yields a slightly different prediction. For $`K`$ stochastic passes:
   $`f_\theta^{(k)}(\mathbf{x}), \quad k=1,\dots,K.`$
   We estimate:
   $`\mu(\mathbf{x}) = \frac{1}{K}\sum_{k=1}^K f_\theta^{(k)}(\mathbf{x}) \quad\text{and}\quad \sigma(\mathbf{x}) = \sqrt{\frac{1}{K}\sum_{k=1}^K (f_\theta^{(k)}(\mathbf{x}) - \mu(\mathbf{x}))^2}.`$

   This provides a distribution-like estimate of model uncertainty, similar to a posterior in Bayesian methods.

2. **Acquisition Function (UCB):**
   We use an Upper Confidence Bound (UCB) acquisition function to choose new configurations:
   $`\text{UCB}(\mathbf{x}) = \mu(\mathbf{x}) + \beta \sigma(\mathbf{x}),`$
   where $`\beta > 0`$ trades off exploration ($`\sigma(\mathbf{x})`$) and exploitation ($`\mu(\mathbf{x})`$). This guides us to either reduce uncertainty by exploring unknown regions or capitalize on known promising areas.

### Efficiency and Complexity

If each parameter had $`n`$ discrete values, exhaustive search takes $`O(n^D)`$ evaluations, which explodes combinatorially. By using the learned model and the UCB criterion, we only evaluate $`T \ll n^D`$ configurations in practice. Each evaluation updates our surrogate, making subsequent steps more informed. Thus, we achieve huge savings in time and computational resources.

### Clustering in Latent Space

The model also maps configurations into a latent space, where:
$`\mathbf{x} \xrightarrow{e_\phi} \mathbf{z} \in \mathbb{R}^L.`$
We cluster these latent representations to identify and focus on high-performing regions. Clustering helps refine where we search next and ensures we don't get stuck or wander aimlessly in the configuration space.

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

## Implementation and Object Details

### Objects and Their Roles

- **`ConfigOptimizer` (Main Class):**  
  Coordinates the entire process:
  - Extracts configuration boundaries from `TextExtractor`.
  - Normalizes parameters to $`[0,1]^D`$.
  - Manages the meta-learning model and training buffer.
  - Employs acquisition functions (UCB) and clustering for guided search.
  - Saves and reloads states, ensuring progress persistence.

- **`MetaLearningModel`:**  
  A neural model composed of:
  - **`ConfigEncoder`:** Maps $`\mathbf{x}$ to $\mathbf{z}`$, a compressed representation capturing essential features.
  - **`PerformancePredictor`:** Maps $`\mathbf{z}`$ to a predicted score, $`\hat{y}`$.
  
  Together:
  $`f_\theta(\mathbf{x}) = p_\psi(e_\phi(\mathbf{x})).`$

- **Clustering (`config_clusters`):**  
  Maintains a set of previously evaluated configurations in latent space. Clusters guide the initial suggestions, ensuring both stable exploitation and creative exploration.

- **`OptimizationRecord`, `ClusterMember`, `OCRResult`:**  
  Data classes used to store evaluations, best configurations, OCR outputs, and cluster information. They maintain a historical record that supports retraining and analysis.

- **`extract_config_space()` and Normalization:**  
  Extracts parameter ranges from the pipeline and normalizes them. Each parameter originally in $`[a,b]`$ is scaled to $`[0,1]`$:
  ` x_{\text{norm}} = \frac{x - a}{b - a}.`$

- **`vector_to_config_dictionary()`:**  
  Converts the normalized vector $`\mathbf{x}`$ back into a human-readable dictionary of steps and parameters, ensuring transparency and reproducibility of the final results.

- **`predict_with_uncertainty()`:**  
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
