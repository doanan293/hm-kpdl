# Regression Homework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a custom Linear Regression model from scratch, train it, compare it with scikit-learn's `LinearRegression` model using the Boston Housing dataset, and evaluate RMSE on the test set.

**Architecture:** Use a class-based approach `MyLinearRegression` matching scikit-learn's `.fit()` and `.predict()` signatures. Use pseudo-inverse closed-form math ($w = (X^T X)^{-1} X^T y$) for numerical stability. Parse the local `BostonDataset.txt` file directly to remain robust against the scikit-learn >= 1.2 `load_boston` deprecation.

**Tech Stack:** Python 3.11, NumPy, Pandas, Scikit-Learn, Jupyter Notebook.

---

### Task 1: Create local verification test script

**Files:**
- Create: `/home/andv/personal/hm-kpdl/docs/superpowers/scratch/test_regression.py`

- [x] **Step 1: Write a verification script**
  Create a python script to load a dummy regression dataset, train both `MyLinearRegression` and scikit-learn's `LinearRegression`, and compare their coefficients, intercepts, predictions, and RMSE.
  
  Write `/home/andv/personal/hm-kpdl/docs/superpowers/scratch/test_regression.py`:
  ```python
  import numpy as np
  import pandas as pd
  import math
  from sklearn import datasets, linear_model
  from sklearn.metrics import mean_squared_error

  class MyLinearRegression:
      def __init__(self):
          self.w = None
          self.coef_ = None
          self.intercept_ = None

      def fit(self, X, y):
          X_b = np.hstack((np.ones((X.shape[0], 1)), X))
          # Compute weights using pseudo-inverse for numerical stability
          self.w = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
          self.intercept_ = self.w[0]
          self.coef_ = self.w[1:]
          return self

      def predict(self, X):
          X_b = np.hstack((np.ones((X.shape[0], 1)), X))
          return X_b.dot(self.w)

  def test_regression_implementation():
      # Generate simple toy dataset
      np.random.seed(42)
      X = np.random.rand(100, 5)
      y = X.dot(np.array([1.5, -2.0, 3.0, 0.5, -1.0])) + 2.5 + np.random.randn(100) * 0.1

      # 1. Train Sklearn
      model_sk = linear_model.LinearRegression()
      model_sk.fit(X, y)

      # 2. Train MyLinearRegression
      model_my = MyLinearRegression()
      model_my.fit(X, y)

      # 3. Asserts for weights equivalence
      print("Testing weights equivalence...")
      assert np.allclose(model_sk.intercept_, model_my.intercept_), f"Intercept mismatch: {model_sk.intercept_} vs {model_my.intercept_}"
      assert np.allclose(model_sk.coef_, model_my.coef_), f"Coef mismatch: {model_sk.coef_} vs {model_my.coef_}"
      print("Weights are perfectly equivalent!")

      # 4. Asserts for predictions equivalence
      X_test = np.random.rand(20, 5)
      pred_sk = model_sk.predict(X_test)
      pred_my = model_my.predict(X_test)
      assert np.allclose(pred_sk, pred_my), "Predictions mismatch!"
      print("Predictions are perfectly equivalent!")

      # 5. Asserts for RMSE
      rmse_sk = math.sqrt(mean_squared_error(y[:20], pred_sk))
      rmse_my = math.sqrt(mean_squared_error(y[:20], pred_my))
      assert np.allclose(rmse_sk, rmse_my), "RMSE mismatch!"
      print("RMSE metrics are perfectly equivalent!")
      print("All tests passed successfully!")

  if __name__ == "__main__":
      test_regression_implementation()
  ```

- [x] **Step 2: Run verification script**
  Run: `python3 /home/andv/personal/hm-kpdl/docs/superpowers/scratch/test_regression.py`
  Expected output: "All tests passed successfully!"

---

### Task 2: Populate solution cells in Jupyter Notebook

**Files:**
- Modify: `3-Regression/Homework/Regression_Homework.ipynb`

- [x] **Step 1: Replace template with solution notebook**
  Use a Python script to robustly parse the JSON of `Regression_Homework.ipynb`, inject modern local file parsing for `BostonDataset.txt` to avoid `load_boston()` ImportError, and append fully coded solution cells.

  Code of solution cells to inject:
  ```json
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Khởi tạo đối tượng giả lập để tương thích ngược với cấu trúc dataset.data và dataset.target\n",
        "class BostonDataset:\n",
        "    def __init__(self, data, target):\n",
        "        self.data = data\n",
        "        self.target = target\n",
        "\n",
        "# Đọc dữ liệu từ file BostonDataset.txt cục bộ (do datasets.load_boston() đã bị gỡ bỏ ở scikit-learn mới)\n",
        "import os\n",
        "file_path = \"BostonDataset.txt\"\n",
        "if not os.path.exists(file_path):\n",
        "    file_path = os.path.join(\"3-Regression\", \"Homework\", \"BostonDataset.txt\")\n",
        "\n",
        "raw_df = pd.read_csv(file_path, sep=\"\\\\s+\", skiprows=22, header=None)\n",
        "data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])\n",
        "target = raw_df.values[1::2, 2]\n",
        "dataset = BostonDataset(data, target)\n",
        "\n",
        "print(\"Số chiều dữ liệu input: \", dataset.data.shape)\n",
        "print(\"Số chiều dữ liệu target: \", dataset.target.shape)\n",
        "print()\n",
        "\n",
        "print(\"5 mẫu dữ liệu đầu tiên:\")\n",
        "print(\"input: \", dataset.data[:5])\n",
        "print(\"target: \", dataset.target[:5])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Khởi tạo mô hình Linear Regression của thư viện scikit-learn\n",
        "regr = linear_model.LinearRegression()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "class MyLinearRegression:\n",
        "    def __init__(self):\n",
        "        self.w = None\n",
        "        self.coef_ = None\n",
        "        self.intercept_ = None\n",
        "\n",
        "    def fit(self, X, y):\n",
        "        # Thêm cột bias (x0 = 1) vào đầu ma trận đặc trưng X\n",
        "        X_b = np.hstack((np.ones((X.shape[0], 1)), X))\n",
        "        \n",
        "        # Tính nghiệm đóng sử dụng giả nghịch đảo để đảm bảo tính ổn định số học\n",
        "        self.w = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)\n",
        "        \n",
        "        # Lưu trữ intercept_ và coef_\n",
        "        self.intercept_ = self.w[0]\n",
        "        self.coef_ = self.w[1:]\n",
        "        return self\n",
        "\n",
        "    def predict(self, X):\n",
        "        # Thêm cột bias (x0 = 1) trước khi dự đoán\n",
        "        X_b = np.hstack((np.ones((X.shape[0], 1)), X))\n",
        "        return X_b.dot(self.w)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Kiểm tra mô hình tự viết trên một tập dữ liệu nhỏ ngẫu nhiên\n",
        "test_X = np.array([[1, 2], [3, 4], [5, 6]])\n",
        "test_y = np.array([5, 11, 17]) # Quan hệ tuyến tính thực tế: y = 3*x1 + 1*x2 + 0\n",
        "test_model = MyLinearRegression()\n",
        "test_model.fit(test_X, test_y)\n",
        "print(\"Bộ trọng số tự tính:\", test_model.w)\n",
        "print(\"Dự đoán mẫu mới [2, 3]:\", test_model.predict(np.array([[2, 3]])))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Huấn luyện mô hình của thư viện\n",
        "regr.fit(dataset_X_train, dataset_y_train)\n",
        "print(\"Sklearn Intercept (w0):\", regr.intercept_)\n",
        "print(\"Sklearn Coefficients (w1...wn):\", regr.coef_)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Huấn luyện mô hình tự viết\n",
        "my_regr = MyLinearRegression()\n",
        "my_regr.fit(dataset_X_train, dataset_y_train)\n",
        "print(\"My Intercept (w0):\", my_regr.intercept_)\n",
        "print(\"My Coefficients (w1...wn):\", my_regr.coef_)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Dự đoán trên tập kiểm thử bằng mô hình sklearn\n",
        "y_pred_sk = regr.predict(dataset_X_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Dự đoán trên tập kiểm thử bằng mô hình tự viết và so sánh kết quả trong DataFrame\n",
        "y_pred_my = my_regr.predict(dataset_X_test)\n",
        "\n",
        "df_compare = pd.DataFrame({\n",
        "    \"Actual\": dataset_y_test,\n",
        "    \"Sklearn_Predict\": y_pred_sk,\n",
        "    \"My_Predict\": y_pred_my\n",
        "})\n",
        "df_compare.head(20)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Tính toán sai số RMSE trên tập kiểm thử đối với mô hình thư viện\n",
        "rmse_sk = math.sqrt(mean_squared_error(dataset_y_test, y_pred_sk))\n",
        "print(\"RMSE Sklearn: \", rmse_sk)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Tính toán sai số RMSE trên tập kiểm thử đối với mô hình tự viết và so sánh\n",
        "rmse_my = math.sqrt(mean_squared_error(dataset_y_test, y_pred_my))\n",
        "print(\"RMSE My Model: \", rmse_my)\n",
        "print(\"Độ lệch RMSE giữa 2 mô hình:\", abs(rmse_sk - rmse_my))"
      ]
    }
  ```

- [x] **Step 2: Run and verify the notebook**
  Verify the notebook JSON structure is valid by executing a loader script that sequentially runs the code of all notebook cells, verifying that the entire notebook finishes executing with ZERO runtime errors.
