# Linear and Ridge Regression Practice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Solve all five practical programming exercises in the `Regression_Practice_1.ipynb` notebook and verify they run successfully and produce correct outputs.

**Architecture:** Edit target code cells in the JSON notebook structure using `replace_file_content` to add implementations for: Model initialization, training & weight printing, manual vs. library predictions comparison, hyperparameter grid search, and distribution visualization/statistics.

**Tech Stack:** Python 3.11, scikit-learn, numpy, pandas, matplotlib, seaborn.

---

### Task 1: Ridge Regression Initialization
**Files:**
- Modify: `3-Regression/Practice/Practice1/Regression_Practice_1.ipynb` (lines 220-232)

- [ ] **Step 1: Write the minimal implementation to initialize Ridge Regression with alpha = 0.1**
  
  Use `replace_file_content` to update the target cell.
  
  Target Content:
  ```json
      {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {
          "id": "div0BVPsmnFT"
        },
        "outputs": [],
        "source": [
          "##### exercise #####\n",
          "# Yêu cầu: Cài đặt mô hình Ridge Regression với alpha = 0.1\n",
          "# Gợi ý: xem hướng dẫn tại https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html\n",
          "######################"
        ]
      },
  ```

  Replacement Content:
  ```json
      {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {
          "id": "div0BVPsmnFT"
        },
        "outputs": [],
        "source": [
          "##### exercise #####\n",
          "# Yêu cầu: Cài đặt mô hình Ridge Regression với alpha = 0.1\n",
          "# Gợi ý: xem hướng dẫn tại https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html\n",
          "######################\n",
          "regr_ridge = linear_model.Ridge(alpha=0.1)"
        ]
      },
  ```

- [ ] **Step 2: Commit changes**
  
  Run:
  ```bash
  git add 3-Regression/Practice/Practice1/Regression_Practice_1.ipynb
  git commit -m "feat: initialize Ridge Regression model with alpha=0.1"
  ```

---

### Task 2: Ridge Regression Training & Weight Inspection
**Files:**
- Modify: `3-Regression/Practice/Practice1/Regression_Practice_1.ipynb` (lines 261-273)

- [ ] **Step 1: Write implementation to train Ridge Regression model and print parameters**

  Use `replace_file_content` to update the target cell.

  Target Content:
  ```json
      {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {
          "id": "zQqcfUc2mnFV"
        },
        "outputs": [],
        "source": [
          "##### exercise #####\n",
          "# Yêu cầu: Huấn luyện mô hình Ridge Regression và in ra các trọng số w0, w1, ...,wn của mô hình\n",
          "# Gợi ý: xem hướng dẫn tại https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html\n",
          "######################"
        ]
      },
  ```

  Replacement Content:
  ```json
      {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {
          "id": "zQqcfUc2mnFV"
        },
        "outputs": [],
        "source": [
          "##### exercise #####\n",
          "# Yêu cầu: Huấn luyện mô hình Ridge Regression và in ra các trọng số w0, w1, ...,wn của mô hình\n",
          "# Gợi ý: xem hướng dẫn tại https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html\n",
          "######################\n",
          "regr_ridge.fit(diabetes_X_train, diabetes_y_train)\n",
          "print(\"[w1, ... w_n] = \", regr_ridge.coef_)\n",
          "print(\"w0 = \", regr_ridge.intercept_)"
        ]
      },
  ```

- [ ] **Step 2: Commit changes**
  
  Run:
  ```bash
  git add 3-Regression/Practice/Practice1/Regression_Practice_1.ipynb
  git commit -m "feat: train Ridge model and output intercept and coefficients"
  ```

---

### Task 3: Manual vs. Library Predictions Comparison
**Files:**
- Modify: `3-Regression/Practice/Practice1/Regression_Practice_1.ipynb` (lines 275-306)

- [ ] **Step 1: Write implementation for manual predictions comparison**

  Use `replace_file_content` to update the target cell.

  Target Content:
  ```json
      {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {
          "id": "Q4jrTHP_mnFV",
          "tags": []
        },
        "outputs": [],
        "source": [
          "##### exercise #####\n",
          "# Yêu cầu: tính giá trị dự đoán của mô hình trên mẫu đầu tiên của tập test và so sánh với kết quả của thư viện\n",
          "# Gợi ý: sử dụng công thức y = w0 + w1*x1 + w1*x2 + ... + w_n*x_n\n",
          "######################\n",
          "# Dự đoán thử cho trường hợp đầu tiên\n",
          "\n",
          "# Giá trị đúng\n",
          "print(\"Gia tri true: \", diabetes_y_test[0])\n",
          "\n",
          "# Dự đoán cho mô hình Linear Regression sử dụng hàm dự đoán của thư viện\n",
          "y_pred_linear = regr.predict(diabetes_X_test[0:1])\n",
          "print(\"Gia tri du doan cho mô hình linear regression: \", y_pred_linear)\n",
          "\n",
          "# Viết code tính và in kết quả dự đoán cho mô hình Linear Regression sử dụng công thức tại đây\n",
          "\n",
          "# Dự đoán cho mô hình Ridge Regression sử dụng hàm dự đoán của thư viện\n",
          "y_pred_ridge = regr_ridge.predict(diabetes_X_test[0:1])\n",
          "print(\"Gia tri du doan cho mô hình ridge regression: \", y_pred_ridge)\n",
          "\n",
          "# Viết code tính và in kết quả dự đoán cho mô hình Ridge Regression sử dụng công thức tại đây\n",
          "\n",
          "######################"
        ]
      },
  ```

  Replacement Content:
  ```json
      {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {
          "id": "Q4jrTHP_mnFV",
          "tags": []
        },
        "outputs": [],
        "source": [
          "##### exercise #####\n",
          "# Yêu cầu: tính giá trị dự đoán của mô hình trên mẫu đầu tiên của tập test và so sánh với kết quả của thư viện\n",
          "# Gợi ý: sử dụng công thức y = w0 + w1*x1 + w1*x2 + ... + w_n*x_n\n",
          "######################\n",
          "# Dự đoán thử cho trường hợp đầu tiên\n",
          "\n",
          "# Giá trị đúng\n",
          "print(\"Gia tri true: \", diabetes_y_test[0])\n",
          "\n",
          "# Dự đoán cho mô hình Linear Regression sử dụng hàm dự đoán của thư viện\n",
          "y_pred_linear = regr.predict(diabetes_X_test[0:1])\n",
          "print(\"Gia tri du doan cho mô hình linear regression: \", y_pred_linear)\n",
          "\n",
          "# Viết code tính và in kết quả dự đoán cho mô hình Linear Regression sử dụng công thức tại đây\n",
          "y_manual_linear = regr.intercept_ + np.dot(diabetes_X_test[0], regr.coef_)\n",
          "print(\"Gia tri du doan thu cong cho mo hinh linear regression: \", y_manual_linear)\n",
          "\n",
          "# Dự đoán cho mô hình Ridge Regression sử dụng hàm dự đoán của thư viện\n",
          "y_pred_ridge = regr_ridge.predict(diabetes_X_test[0:1])\n",
          "print(\"Gia tri du doan cho mô hình ridge regression: \", y_pred_ridge)\n",
          "\n",
          "# Viết code tính và in kết quả dự đoán cho mô hình Ridge Regression sử dụng công thức tại đây\n",
          "y_manual_ridge = regr_ridge.intercept_ + np.dot(diabetes_X_test[0], regr_ridge.coef_)\n",
          "print(\"Gia tri du doan thu cong cho mo hinh ridge regression: \", y_manual_ridge)\n",
          "\n",
          "######################"
         ]
      },
  ```

- [ ] **Step 2: Commit changes**
  
  Run:
  ```bash
  git add 3-Regression/Practice/Practice1/Regression_Practice_1.ipynb
  git commit -m "feat: compare manual matrix-multiply predictions vs sklearn model output"
  ```

---

### Task 4: Hyperparameter Grid Search & RMSE Evaluation
**Files:**
- Modify: `3-Regression/Practice/Practice1/Regression_Practice_1.ipynb` (lines 367-386)

- [ ] **Step 1: Write implementation for Hyperparameter Grid Search**

  Use `replace_file_content` to update the target cell.

  Target Content:
  ```json
      {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {
          "id": "JETkYN9omnFX"
        },
        "outputs": [],
        "source": [
          "##### exercise #####\n",
          "# Yêu cầu: đánh giá độ đo RMSE của mô hình Ridge Regression với các hằng số phạt khác nhau, in ra kết quả.\n",
          "# Gợi ý: Các bước làm:\n",
          "# - Lặp theo danh sách các hằng số phạt\n",
          "# - Dựng các mô hình Ridge Regression với mỗi hằng số phạt tương ứng\n",
          "# - Huấn luyện các mô hình và dự đoán\n",
          "# - Tính RMSE tương ứng\n",
          "######################\n",
          "\n",
          "# Các giá trị hằng số phạt cho trước\n",
          "_lambda = [0, 0.0001, 0.01, 0.04, 0.05, 0.06, 0.1, 0.5, 1, 5, 10, 20]"
        ]
      },
  ```

  Replacement Content:
  ```json
      {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {
          "id": "JETkYN9omnFX"
        },
        "outputs": [],
        "source": [
          "##### exercise #####\n",
          "# Yêu cầu: đánh giá độ đo RMSE của mô hình Ridge Regression với các hằng số phạt khác nhau, in ra kết quả.\n",
          "# Gợi ý: Các bước làm:\n",
          "# - Lặp theo danh sách các hằng số phạt\n",
          "# - Dựng các mô hình Ridge Regression với mỗi hằng số phạt tương ứng\n",
          "# - Huấn luyện các mô hình và dự đoán\n",
          "# - Tính RMSE tương ứng\n",
          "######################\n",
          "\n",
          "# Các giá trị hằng số phạt cho trước\n",
          "_lambda = [0, 0.0001, 0.01, 0.04, 0.05, 0.06, 0.1, 0.5, 1, 5, 10, 20]\n",
          "\n",
          "for l in _lambda:\n",
          "    model = linear_model.Ridge(alpha=l)\n",
          "    model.fit(diabetes_X_train, diabetes_y_train)\n",
          "    y_pred = model.predict(diabetes_X_test)\n",
          "    rmse = math.sqrt(mean_squared_error(diabetes_y_test, y_pred))\n",
          "    print(f\"Alpha (lambda) = {l:7.4f} | RMSE = {rmse:.4f}\")"
        ]
      },
  ```

- [ ] **Step 2: Commit changes**
  
  Run:
  ```bash
  git add 3-Regression/Practice/Practice1/Regression_Practice_1.ipynb
  git commit -m "feat: implement loop evaluating Ridge Regression RMSE across alphas"
  ```

---

### Task 5: Stats & Distribution Visualization
**Files:**
- Modify: `3-Regression/Practice/Practice1/Regression_Practice_1.ipynb` (lines 434-447)

- [ ] **Step 1: Write implementation for statistical analysis and visualization of predictions**

  Use `replace_file_content` to update the target cell.

  Target Content:
  ```json
      {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {
          "id": "DET-Jf6_mnFZ",
          "tags": []
        },
        "outputs": [],
        "source": [
          "##### exercise #####\n",
          "# Yêu cầu: Tính các chỉ số thống kê và vẽ biểu đồ phân phối của chỉ số dự đoán bằng mô hình Linear Regression, quan sát và nhận xét\n",
          "# Gợi ý: sử dụng sns và pd\n",
          "######################"
        ]
      },
  ```

  Replacement Content:
  ```json
      {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {
          "id": "DET-Jf6_mnFZ",
          "tags": []
        },
        "outputs": [],
        "source": [
          "##### exercise #####\n",
          "# Yêu cầu: Tính các chỉ số thống kê và vẽ biểu đồ phân phối của chỉ số dự đoán bằng mô hình Linear Regression, quan sát và nhận xét\n",
          "# Gợi ý: sử dụng sns và pd\n",
          "######################\n",
          "import seaborn as sns\n",
          "sns.distplot(diabetes_y_pred)\n",
          "plt.title(\"Bieu do phan phoi chi so du doan cua mo hinh Linear Regression\")\n",
          "plt.show()\n",
          "pd.DataFrame(data=diabetes_y_pred, columns=[\"values\"]).describe()"
        ]
      },
  ```

- [ ] **Step 2: Commit changes**
  
  Run:
  ```bash
  git add 3-Regression/Practice/Practice1/Regression_Practice_1.ipynb
  git commit -m "feat: visualize Linear Regression predictions and compute description statistics"
  ```

---

### Task 6: Execution & Verification
**Files:**
- Create: `scratch/run_notebook.py`
- Run validation script to ensure notebook runs successfully.

- [ ] **Step 1: Create a lightweight python script to execute notebook cells**

  Create `scratch/run_notebook.py` using `write_to_file`. The script will parse the `.ipynb` file, filter out shell commands like `!pip`, configure `matplotlib` to run in headless `Agg` mode, mock `plt.show()` to avoid blocking, and execute all cells in sequence.

  ```python
  import json
  import sys
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  plt.show = lambda *args, **kwargs: None

  notebook_path = "3-Regression/Practice/Practice1/Regression_Practice_1.ipynb"
  with open(notebook_path, "r", encoding="utf-8") as f:
      nb = json.load(f)

  global_env = {}
  cell_idx = 0
  for cell in nb["cells"]:
      if cell["cell_type"] == "code":
          cell_idx += 1
          source = "".join(cell["source"])
          lines = []
          for line in source.splitlines():
              if line.strip().startswith("!"):
                  continue
              lines.append(line)
          code = "\n".join(lines)
          if not code.strip():
              continue
          print(f"--- Executing Cell {cell_idx} ---")
          try:
              exec(code, global_env)
          except Exception as e:
              print(f"Error in Cell {cell_idx}: {e}")
              sys.exit(1)

  print("Notebook executed successfully without errors!")
  ```

- [ ] **Step 2: Run the notebook validator**

  Run:
  ```bash
  uv run python scratch/run_notebook.py
  ```
  Expected output: "Notebook executed successfully without errors!" and all metrics and stats printed.

- [ ] **Step 3: Clean up verification script**

  Delete `scratch/run_notebook.py`.
