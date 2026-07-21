# Day 2 – July 21, 2026

## Topic
Multi-Layer Perceptron (MLP) for Binary Classification – Predicting Customer Churn

## Key Formulas & Concepts

**Activation Functions:**
- **ReLU (Hidden Layers):** `f(x) = max(0, x)`
- **Sigmoid (Output Layer):** `σ(x) = 1 / (1 + e^(-x))`

**Loss Function:**
- **Binary Cross-Entropy:** `L(y, ŷ) = -[y * log(ŷ) + (1 - y) * log(1 - ŷ)]`

**Model Architecture:**
- Input Layer: 11 features
- Hidden Layer 1: 11 neurons (ReLU)
- Hidden Layer 2: 7 neurons (ReLU)
- Output Layer: 1 neuron (Sigmoid)

**Optimizer:**
- **Adam (Adaptive Moment Estimation):** An extension of stochastic gradient descent that maintains per-parameter learning rates.

---
## Results
### Dataset: Customer Churn Prediction

**Source:** [Churn_Modelling.csv](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)

**Data Preparation:**
1.  **Feature Selection:** Dropped non-informative columns `RowNumber`, `CustomerId`, and `Surname`.
2.  **Encoding:** One-hot encoded `Geography` and `Gender` using `pandas.get_dummies(drop_first=True)`. This created binary columns for `Geography_Germany`, `Geography_Spain`, and `Gender_Male`.
3.  **Target Variable:** `Exited` (1 for churn, 0 for retained).

**Data Splitting & Preprocessing:**
- **Split:** 80% training (8,000 samples), 20% test (2,000 samples).
- **Standardization:** Applied `StandardScaler` to normalize features (zero mean, unit variance). This is crucial for the faster convergence of gradient-based optimizers like Adam.

**Training & Performance:**
- **Epochs:** 100
- **Validation Split:** 20% of training data.
- **Test Accuracy:** Achieved `85.25%` on the held-out test set.
- **Learning Curves:** The model's training and validation loss converged smoothly, indicating stable learning without significant overfitting. Training accuracy steadily increased and plateaued, while validation accuracy remained consistent.

**Model Analysis:**
- **Parameter Count:** Total of 224 trainable parameters. This relatively small model is computationally efficient while still capturing the non-linear relationships in the data.
- **Weight Visualization:** Post-training, we can access the model's weights and biases for each layer, providing insight into how the network transforms inputs.

### Comparison with Theoretical Concepts
- **Overcoming XOR Limitation:** The inclusion of a hidden layer allows the MLP to learn non-linear decision boundaries. This successfully addresses the structural limitation of the single-layer perceptron demonstrated on Day 1, which was unable to model the XOR problem.

---

## Challenges & Limitations Faced

### 1. Data Standardization (Scaler Fitting)
- Forgetting to use `scaler.fit_transform()` on the training set and `scaler.transform()` on the test set. Using `fit_transform` on both would leak information from the test set into the model.


### 2. Categorical Feature Encoding
- Applying one-hot encoding without setting `drop_first=True` can lead to the "dummy variable trap" where one feature can be predicted from the others, causing multicollinearity and making the model unstable.


---

## Summary

I Learn first how this pipeline actually work the goal for this exercise is not to achive the best accuracy. 

