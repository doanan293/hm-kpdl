# Design Spec: Linear and Ridge Regression Practice

## Objective
Solve all five practical programming exercises in the `Regression_Practice_1.ipynb` notebook, focusing on Ridge Regression implementation, weight inspection, manual vs. library predictions, hyperparameter evaluation, and statistical/visual distribution analysis.

## Proposed Design

### 1. Ridge Regression Model Initialization
- **Cell**: Under `## Xây dựng mô hình Regression sử dụng Sklearn`
- **Implementation**:
  ```python
  regr_ridge = linear_model.Ridge(alpha=0.1)
  ```

### 2. Ridge Regression Training and Parameter Extraction
- **Cell**: Under `## Training mô hình`
- **Implementation**:
  ```python
  regr_ridge.fit(diabetes_X_train, diabetes_y_train)
  print("[w1, ... w_n] = ", regr_ridge.coef_)
  print("w0 = ", regr_ridge.intercept_)
  ```

### 3. Manual vs. Library Prediction Comparison
- **Cell**: Under `## Training mô hình` (exercise cell for manual prediction comparison)
- **Implementation**:
  - Manual Linear Regression prediction using the dot product formula:
    ```python
    y_manual_linear = regr.intercept_ + np.dot(diabetes_X_test[0], regr.coef_)
    print("Gia tri du doan thu cong cho mo hinh linear regression: ", y_manual_linear)
    ```
  - Manual Ridge Regression prediction:
    ```python
    y_manual_ridge = regr_ridge.intercept_ + np.dot(diabetes_X_test[0], regr_ridge.coef_)
    print("Gia tri du doan thu cong cho mo hinh ridge regression: ", y_manual_ridge)
    ```

### 4. Hyperparameter Alpha (Lambda) Grid Search & Evaluation
- **Cell**: Under `## Đánh giá`
- **Implementation**:
  - Iterate through `_lambda` values.
  - Instantiate and train `linear_model.Ridge(alpha=l)`.
  - Compute the RMSE of predictions on the test set and print the result:
    ```python
    for l in _lambda:
        model = linear_model.Ridge(alpha=l)
        model.fit(diabetes_X_train, diabetes_y_train)
        y_pred = model.predict(diabetes_X_test)
        rmse = math.sqrt(mean_squared_error(diabetes_y_test, y_pred))
        print(f"Alpha (lambda) = {l:7.4f} | RMSE = {rmse:.4f}")
    ```

### 5. Statistics & Distribution Visualization for Linear Regression Predictions
- **Cell**: Under `### Vẽ biểu đồ phân phối cho chỉ số dự đoán của mô hình linear regression`
- **Implementation**:
  - Draw the distribution plot using `sns.distplot` (or `sns.histplot(..., kde=True)` for compatibility).
  - Print the statistics of predicted values using `pd.DataFrame.describe()`:
    ```python
    import seaborn as sns
    sns.distplot(diabetes_y_pred)
    plt.title("Bieu do phan phoi chi so du doan cua mo hinh Linear Regression")
    plt.show()
    
    pd.DataFrame(data=diabetes_y_pred, columns=["values"]).describe()
    ```

## Verification Plan
- Verify that running the notebook sequentially works without any errors.
- Ensure manual predictions exactly match the library predictions (`y_pred_linear` and `y_pred_ridge` up to floating-point precision).
- Ensure RMSE values are correctly printed for each $\alpha$ value.
- Ensure the distribution plot displays correctly and the statistical analysis of predicted values matches expected ranges.
