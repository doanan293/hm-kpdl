# Design Spec: Solutions for Regression Practice 2

This document describes the design and solution implementation for the exercises in the Jupyter Notebook `Regression_Practice_2.ipynb`.

## User Review Required

No critical architectural changes or security vulnerabilities are introduced. The solutions are self-contained python code cells that follow standard machine learning steps in Scikit-Learn.

## Proposed Changes

### [Component] Jupyter Notebook Exercises

We will modify `/home/andv/personal/hm-kpdl/3-Regression/Practice/Practice2/Regression_Practice_2.ipynb` by inserting the correct solutions into the four designated exercise cells.

#### [MODIFY] [Regression_Practice_2.ipynb](file:///home/andv/personal/hm-kpdl/3-Regression/Practice/Practice2/Regression_Practice_2.ipynb)

* **Exercise 1 (Sorting):** Chronologically sort the data by `Year` and `Month` and reset the index to prevent indexing issues during slicing.
  ```python
  df = df.sort_values(by=['Year', 'Month']).reset_index(drop=True)
  ```

* **Exercise 2 (Training):** Construct the `LinearRegression` model and fit it on the scaled `X_train` features and `y_train` labels.
  ```python
  from sklearn.linear_model import LinearRegression
  model1 = LinearRegression()
  model1.fit(X_train, y_train)
  ```

* **Exercise 3 (Plotting):** Generate a matplotlib line plot comparing the actual target values `y_test` and predicted values `y_pred_test`.
  ```python
  plt.figure(figsize=(9, 6))
  plt.plot(y_test, label='Thực tế (y_test)', color='blue', marker='o')
  plt.plot(y_pred_test, label='Dự đoán (y_pred_test)', color='red', linestyle='--', marker='x')
  plt.xlabel('Time index')
  plt.ylabel('Sales')
  plt.title('So sánh doanh thu thực tế và doanh thu dự đoán')
  plt.legend()
  plt.show()
  ```

* **Exercise 4 (Feature Merging):** Horizontally stack the one-hot encoded `Month` features with the scaled training features `X_train`.
  ```python
  X_train = np.hstack((X_train, month_onehot_train.values))
  ```

## Verification Plan

We will run a validation script (or execute the notebook) to verify:
1. All notebook cells compile and execute without errors.
2. The model achieves the expected standard output metrics (RMSE and Mean relative error values).
