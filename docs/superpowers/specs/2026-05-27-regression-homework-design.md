# Design Specification: Regression Homework Solver

This document outlines the design specification for completing the machine learning regression homework assignment in [Regression_Homework.ipynb](file:///home/andv/personal/hm-kpdl/3-Regression/Homework/Regression_Homework.ipynb).

## 1. Objectives

- Implement a custom Linear Regression model from scratch using the Normal Equation / closed-form solution.
- Fit and compare the custom model against scikit-learn's `LinearRegression` model using the Boston Housing dataset.
- Validate and verify that both models yield mathematically equivalent weights (coefficients and intercepts), predictions, and test metrics (RMSE).

---

## 2. Model Architecture

### Custom Class: `MyLinearRegression`

To ensure full compatibility with the existing code cells and scikit-learn's API, the custom model will be implemented as a class:

```python
class MyLinearRegression:
    def __init__(self):
        self.w = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        # 1. Add bias column of ones as the first column of X
        X_b = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # 2. Compute weights using the pseudo-inverse Normal Equation: w = (X^T * X)^-1 * X^T * y
        # We use np.linalg.pinv for numerical stability (handles singular matrices)
        self.w = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
        # 3. Store coefficients and intercept
        self.intercept_ = self.w[0]
        self.coef_ = self.w[1:]
        return self

    def predict(self, X):
        # 1. Add bias column of ones
        X_b = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # 2. Calculate predictions: y_pred = X_b * w
        return X_b.dot(self.w)
```

---

## 3. Step-by-Step Implementation Workflow

1. **Environment Setup & Imports**: Keep existing imports (`matplotlib.pyplot`, `numpy`, `pandas`, `math`, `sklearn.datasets`, `linear_model`, `mean_squared_error`, `r2_score`).
2. **Data Loading & Train-Test Split**:
   - Keep current Boston Dataset loading.
   - Keep dataset splitting logic (train size: 404, test size: 101/remaining). Note that the text cell says 362/80 but the actual split indices in the template use `[:404]` and `[405:]`. We will keep the code's splitting index as-is to preserve assignment integrity.
3. **Model Training**:
   - Instantiate and fit `linear_model.LinearRegression()`.
   - Instantiate and fit `MyLinearRegression()`.
4. **Weights Comparison**:
   - Print `intercept_` and `coef_` for both models.
5. **Predictions Comparison**:
   - Call `.predict()` for both models on `dataset_X_test`.
   - Create a pandas DataFrame containing columns: `Actual`, `Sklearn_Predict`, and `My_Predict`.
6. **Evaluation**:
   - Calculate RMSE for both models on `dataset_y_test` and `dataset_X_test`.
   - Print both RMSE values to verify they are equivalent (or nearly equivalent up to floating-point precision).

---

## 4. Spec Self-Review Check

- **Placeholder scan**: No placeholders (TBD/TODO) remain.
- **Internal consistency**: The class design matches standard API expectations, and normal equation calculations align with class theory.
- **Scope check**: Focuses exactly on completing the specified homework sections in the notebook.
- **Ambiguity check**: The choice of Normal Equation vs Gradient Descent is explicitly clarified as Normal Equation.
