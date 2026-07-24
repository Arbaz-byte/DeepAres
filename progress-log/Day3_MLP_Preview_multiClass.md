# Day 3 – July 23, 2026

## Topic
Multi-Layer Perceptron (MLP) for Binary and Multiclass Classification

## Key Concepts & Formulas

### Binary Classification (Customer Churn)

**Activation Functions:**
- **Hidden Layers (ReLU):** `f(x) = max(0, x)`
- **Output Layer (Sigmoid):** `σ(x) = 1 / (1 + e^(-x))`

**Loss Function:**
- **Binary Cross-Entropy:** `L(y, ŷ) = -[y * log(ŷ) + (1 - y) * log(1 - ŷ)]`

**Model Architecture:**
- Input Layer: 11 features
- Hidden Layer 1: 11 neurons (ReLU)
- Hidden Layer 2: 7 neurons (ReLU)
- Output Layer: 1 neuron (Sigmoid)

**Optimizer:** Adam (Adaptive Moment Estimation)

### Multiclass Classification (MNIST Handwritten Digits)

**Softmax Activation:**
$$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}, \quad K = 10 \text{ for digits 0–9}$$

**Loss Function:**
- **Sparse Categorical Cross-Entropy:** `L = -∑_{i=1}^{K} y_i * log(ŷ_i)` where `y_i` is 1 for the true class.

**Model Architecture:**
- Input: 784 (flattened 28×28 image)
- Hidden Layer: 128 neurons (ReLU)
- Output Layer: 10 neurons (Softmax)

---
## Results
### 1. Binary Classification – Customer Churn Prediction

**Dataset:** Churn_Modelling.csv (10,000 records)

**Data Preparation:**
- Dropped non-informative columns (`RowNumber`, `CustomerId`, `Surname`).
- One‑hot encoded `Geography` and `Gender` with `drop_first=True` to avoid multicollinearity.
- Train/Test split: 80% / 20%.
- Standardized features using `StandardScaler` (fit on training set only).

**Training:**
- Epochs: 100, Validation split: 20%
- Optimizer: Adam, Loss: Binary Cross‑Entropy
- **Test Accuracy:** `85.25%`

**Learning Curves:** Both training and validation loss converged smoothly, indicating stable learning with minimal overfitting.

---

### 2. Multiclass Classification – MNIST Handwritten Digits

**Dataset:** Keras MNIST (60k train, 10k test, 28×28 grayscale)

**Data Preparation:**
- Normalized pixel values by dividing by 256 to map to [0, 1).
- Used `Flatten` layer to reshape each image into a 784‑dimensional vector.
- Labels remained as integers (no one‑hot encoding required).

**Training:**
- Epochs: 10, Validation split: 20%
- Optimizer: Adam, Loss: Sparse Categorical Cross‑Entropy
- **Test Accuracy:** `92.25%`

**Misclassification Analysis:**
- Visual inspection of 8 misclassified samples revealed that errors often occur for visually similar digits (e.g., 4 vs. 9, 7 vs. 9, 3 vs. 8, 2 vs. 7). This insight provides a clear direction for further improvement (e.g., data augmentation, more complex architectures).

---

### Comparison with Theoretical Concepts
- **Overcoming XOR Limitation:** Both models included hidden layers, enabling them to learn non‑linear decision boundaries—directly addressing the single‑layer perceptron limitation demonstrated on Day 1.
- **Activation Functions:** Sigmoid for binary output provides a probability; Softmax for multiclass gives a probability distribution across all classes.
- **Loss Functions:** Binary cross‑entropy vs. sparse categorical cross‑entropy – the latter is optimised for integer labels, simplifying the data pipeline.

---

## Challenges & Limitations Faced

### 1. Data Standardization (Binary Classification)
-  Forgetting to `fit` the `StandardScaler` only on the training set and applying `transform` to both train and test sets.


### 2. Categorical Feature Encoding (Binary Classification)
-  One‑hot encoding without `drop_first=True` can cause the dummy variable trap (multicollinearity).

### 3. Threshold Selection for Binary Output
- The sigmoid output is a probability; the default threshold (0.5) may not be optimal for imbalanced datasets or business costs.



---

## Summary
I  successfully implemented MLP pipelines for both binary and multiclass classification tasks. 

- The binary churn model achieved 85% test accuracy, while the multiclass MNIST model achieved 92% test accuracy. 

- Key learnings include the importance of proper data scaling, appropriate activation and loss functions, and the value of visual error analysis. 

- Will try to implement regression algorithm using MLP 




