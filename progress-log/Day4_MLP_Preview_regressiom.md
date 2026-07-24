# Day 4 – July 24, 2026

## Topic
Multi-Layer Perceptron (MLP) for Regression – Predicting Graduate Admission Chances

## Key Formulas & Concepts

**Regression Output Activation:**
- **Linear (Identity):** `f(x) = x` – no transformation, allowing the network to predict any real‑valued target.

**Loss Function:**
- **Mean Squared Error (MSE):** 
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Evaluation Metric:**
- **R² Score (Coefficient of Determination):**
$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$
  - Measures the proportion of variance in the target explained by the model.
  - Closer to 1 indicates better predictive performance.

**Model Architecture:**
- Input: 7 features
- Hidden Layer 1: 7 neurons (ReLU)
- Hidden Layer 2: 10 neurons (ReLU)
- Output Layer: 1 neuron (Linear)

**Optimizer:**
- **Adam:** Adaptive learning rate optimization, suitable for regression tasks with continuous loss.

---
## Results
### Dataset: Graduate Admission Prediction

**Source:** [Admission_Predict.csv](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions)

**Dataset Description:**
- **Samples:** 500
- **Features:** GRE Score, TOEFL Score, University Rating, SOP (Statement of Purpose), LOR (Letter of Recommendation), CGPA, Research (0/1)
- **Target:** Chance of Admit (probability between 0 and 1)

**Data Preparation:**
1.  **Feature Selection:** Dropped `Serial No.` as it is an arbitrary identifier.
2.  **Train/Test Split:** 80% training (400 samples), 20% test (100 samples).
3.  **Scaling:** Applied `MinMaxScaler` to scale all features to the range [0, 1]. This ensures all features contribute equally to gradient updates and helps the optimizer converge faster.

**Training & Performance:**
- **Epochs:** 50
- **Validation Split:** 20% of training data (80 samples)
- **Test R² Score:** Achieved `0.745` on the held-out test set, indicating that the model explains about 74.5% of the variance in admission chances.
- **Loss Curve:** Training and validation MSE decreased steadily and converged, showing stable learning with no significant overfitting.

**Model Analysis:**
- **Parameter Count:** Only 147 trainable parameters, making the model lightweight and efficient.
- **Scaler Choice:** `MinMaxScaler` was used instead of `StandardScaler`. This is appropriate when the data does not follow a Gaussian distribution and when we want to preserve the relative order of values (e.g., GRE scores between 260 and 340).

### Comparison with Previous Tasks
- **Task Shift:** This is a regression problem, unlike the previous classification tasks (churn and MNIST). The target is continuous, so the output layer uses linear activation and MSE loss.
- **Similarities:** The same MLP workflow – data preprocessing, model definition, training, and evaluation – applies across all tasks, demonstrating the versatility of neural networks.

---

## Challenges & Limitations Faced

### 1. Choosing the Right Scaling Method
- **Challenge:** Deciding between `StandardScaler` and `MinMaxScaler`.
- **Observation:** The features (GRE, TOEFL, CGPA, etc.) have known ranges and are not heavily skewed. `MinMaxScaler` is simpler and ensures all values fall in [0,1], which can help with gradient stability.

### 2. Interpreting MSE and R²
- **Challenge:** MSE values are in squared units (e.g., if Chance of Admit is between 0 and 1, MSE is small). It is not directly intuitive.
- **Solution:** The R² score provides a more interpretable metric – it is the proportion of variance explained by the model. An R² of 0.745 indicates a reasonably good fit, but there is room for improvement by adding more features or using a more complex model.


### 3. Model Capacity and Overfitting
- **Challenge:** With only 500 samples, a complex model could overfit. The chosen architecture (7–10–1) is relatively small, with only 147 parameters.
- **Observation:** The training and validation loss curves are close and converge smoothly, suggesting that the model is not overfitting.



---

## Summary
I extended the MLP pipeline to a regression task using the Graduate Admission dataset. 

- The model achieved an R² of 0.745, indicating that it can explain a significant portion of the variance in admission chances. 

**Key learnings include:** 

- The choice of linear activation for regression, the use of MSE loss, and the importance of scaling. 

- The small parameter count and stable learning curves show that neural networks can be effective even with limited data.
