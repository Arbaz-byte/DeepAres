# Day 1 – July 20, 2026

## Topic
Perceptron Algorithm – Batch Vectorized Implementation, Convergence Theorem, and XOR Limitation Analysis

## Key Formula

**Perceptron Hypothesis with Bias Trick:**
$$\hat{y}^{(i)} = \text{sign}(\mathbf{w}^T \mathbf{x}^{(i)})$$

where $\mathbf{w} = [b, w_1, w_2]^T$ and $\mathbf{x}^{(i)} = [1, x_1^{(i)}, x_2^{(i)}]^T$

**Perceptron Loss Function (Continuous Surrogate):**
$$\mathcal{L}(\mathbf{w}) = \max(0, -y^{(i)}(\mathbf{w}^T \mathbf{x}^{(i)}))$$

**Subgradient Update Rule (Vectorized Batch Form):**
$$\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} + \frac{\eta}{M} \sum_{i \in \mathcal{M}(\mathbf{w})} y^{(i)} \mathbf{x}^{(i)}$$

where $\mathcal{M}(\mathbf{w})$ is the set of misclassified samples.

**Convergence Bound (Perceptron Convergence Theorem):**
$$k \leq \left(\frac{R}{\gamma}\right)^2$$

where:
- $R$ = maximum Euclidean norm of input vectors (data radius)
- $\gamma$ = minimum margin (distance from hyperplane)

---
## Results
### Linearly Separable Data (2 Features)

**Data Generation**: make_classification with flip_y=0.01 (1% label noise)
Training Set: 800 samples, Test Set: 200 samples

**Convergence**: Achieved 99.5% accuracy on test set
```
Convergence Bound Analysis:
Data Radius Bound (R): 3.8833
Minimum Margin (gamma): -2.2484 ← Negative margin due to label noise
Theoretical Max Mistake Allowed (K Limit): inf
WARNING: Some margins <= 0 – training did not converge perfectly.
```

### XOR Problem Demonstration
- **Data Generation:** 4 Gaussian clusters arranged in XOR pattern
- **Training Behavior:**
- Loss plateaued after 11 epochs (tolerance-based early stop)
- **Final Accuracy:** Failed to classify correctly
- **Convergence Bound:**
  ```
  Data Radius Bound (R): 6.3392
  Minimum Margin (gamma): -6.2669
  Theoretical Max Mistake Allowed: inf
  ```
- **Visualization:** Attempted to separate non-linear XOR pattern with a single straight line – mathematically impossible, resulting in high error rate

### Comparison with Scikit-learn
- **Custom Perceptron Accuracy:** 99.5%
- **Scikit-learn Perceptron Accuracy:** 99.5%
- **Decision Boundary Overlay:** Both models produced visually identical separating lines

---

## Challenges & Limitations Faced

### 1. Label Noise Masked Convergence
- Used `flip_y=0.01` to introduce 1% label noise, making the data not strictly linearly separable.
- **Observation:** The perceptron could **never reach 0 training errors** due to these intentionally flipped labels.
- **Lesson:** The Convergence Theorem only applies when the data is **perfectly linearly separable**. For noisy data, a maximum epoch limit or tolerance-based early stopping is essential.

### 2. XOR Impossibility Proof
- The algebraic proof showed that no single straight line can separate the four XOR quadrants.
- **Observation:** The perceptron loss plateaued quickly (at epoch 11) and the final decision boundary was arbitrary, confirming the **structural limitation** of single-layer perceptrons.
- **Insight:** This fundamental limitation motivated the development of multi-layer perceptrons (neural networks) and kernel methods.

### 3. Interpretation of Negative Margin
- A negative margin ($\gamma < 0$) indicates that some training points lie on the wrong side of the hyperplane.
- **For linearly separable data with noise:** Negative margin is inevitable – the perceptron minimizes **number of errors**, not margin distance.
- **For XOR data:** Negative margin confirms the data is inherently non-linear.

### 4. Visualizing High-Dimensional Boundaries
- **Challenge:** The visualization tools only support 2D and 3D plotting.
- **Solution:** Added conditional branching to handle datasets with >3 features by printing a message instead of crashing.

### 5. Vectorization vs. Loop-Based Updates
- Initially used a per-sample update loop which was significantly slower.
- **Optimization:** Replaced with matrix-based batch gradient computation, reducing runtime by **~80%** on 1000-sample datasets.
