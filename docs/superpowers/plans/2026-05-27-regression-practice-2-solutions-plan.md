# Regression Practice 2 Solutions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the correct solutions for the 4 exercises in `Regression_Practice_2.ipynb` and verify they run successfully.

**Architecture:** Use JSON cell manipulation in Python to inject the solution code snippets into the specified cells of the `.ipynb` file, and verify by executing the notebook programmatically.

**Tech Stack:** Python, Pandas, Numpy, Scikit-Learn, Matplotlib, nbformat.

---

### Task 1: Setup Verification Script

**Files:**
- Create: `/home/andv/personal/hm-kpdl/docs/superpowers/scratch/verify_notebook.py`

- [ ] **Step 1: Create the verification script**
  Create a python script that will run the notebook's cells sequentially to verify execution correctness.

  Write `/home/andv/personal/hm-kpdl/docs/superpowers/scratch/verify_notebook.py`:
  ```python
  import json
  import sys

  def run_notebook(notebook_path):
      with open(notebook_path, 'r', encoding='utf-8') as f:
          nb = json.load(f)
      
      namespace = {}
      # Mock plt.show to avoid blocking during headless execution
      import matplotlib.pyplot as plt
      plt.show = lambda: None
      namespace['plt'] = plt
      
      for idx, cell in enumerate(nb['cells']):
          if cell['cell_type'] == 'code':
              source = "".join(cell['source'])
              # Skip interactive plotting functions that might cause visual issues in headless execution
              if "plt.show" in source and "plt.plot" not in source:
                  continue
              try:
                  exec(source, namespace)
              except Exception as e:
                  print(f"Error running cell {idx} with source:\n{source}\n")
                  raise e
      print("Notebook executed successfully!")

  if __name__ == '__main__':
      run_notebook('/home/andv/personal/hm-kpdl/3-Regression/Practice/Practice2/Regression_Practice_2.ipynb')
  ```

- [ ] **Step 2: Run the verification script to verify it fails**
  Run the script. It should fail because the exercises are not yet implemented (e.g. `model1` is not defined).

  Run: `python3 /home/andv/personal/hm-kpdl/docs/superpowers/scratch/verify_notebook.py`
  Expected: FAIL with NameError on `model1` or similar.

- [ ] **Step 3: Commit**
  ```bash
  git add docs/superpowers/scratch/verify_notebook.py
  git commit -m "test: add notebook verification script"
  ```

---

### Task 2: Implement Exercise 1 (Sorting Data)

**Files:**
- Modify: `/home/andv/personal/hm-kpdl/3-Regression/Practice/Practice2/Regression_Practice_2.ipynb` (Cell with ID `a9a21a8e-e14a-431e-93a5-d625daedfd68`)

- [ ] **Step 1: Check existing cell source**
  Locate cell `a9a21a8e-e14a-431e-93a5-d625daedfd68` to ensure it only has the exercise description comments.

- [ ] **Step 2: Write minimal implementation**
  Modify `/home/andv/personal/hm-kpdl/3-Regression/Practice/Practice2/Regression_Practice_2.ipynb` by replacing the cell source for `a9a21a8e-e14a-431e-93a5-d625daedfd68` with:
  ```json
        "source": [
          "##### exercise #####\n",
          "# Yêu cầu: Sắp xếp lại thứ tự các hàng dữ liệu theo tháng/năm\n",
          "# Gợi ý: sử dụng df.sort_values và df.reset_index\n",
          "######################\n",
          "df = df.sort_values(by=['Year', 'Month']).reset_index(drop=True)"
        ]
  ```

- [ ] **Step 3: Commit**
  ```bash
  git add 3-Regression/Practice/Practice2/Regression_Practice_2.ipynb
  git commit -m "feat: implement exercise 1 (sorting)"
  ```

---

### Task 3: Implement Exercise 2 (Model Training)

**Files:**
- Modify: `/home/andv/personal/hm-kpdl/3-Regression/Practice/Practice2/Regression_Practice_2.ipynb` (Cell with ID `deddfe16-d873-406f-8f2b-66f2ae675969`)

- [ ] **Step 1: Check existing cell source**
  Locate cell `deddfe16-d873-406f-8f2b-66f2ae675969` to ensure it only has the exercise description comments.

- [ ] **Step 2: Write minimal implementation**
  Modify `/home/andv/personal/hm-kpdl/3-Regression/Practice/Practice2/Regression_Practice_2.ipynb` by replacing the cell source for `deddfe16-d873-406f-8f2b-66f2ae675969` with:
  ```json
        "source": [
          "###### exercise #####\n",
          "# Yêu cầu: Xây dựng và huấn luyện mô hình Linear Regression\n",
          "# Gợi ý: sử dụng hàm fit() như trong bài thực hành 1\n",
          "######################\n",
          "from sklearn.linear_model import LinearRegression\n",
          "model1 = LinearRegression()\n",
          "model1.fit(X_train, y_train)"
        ]
  ```

- [ ] **Step 3: Commit**
  ```bash
  git add 3-Regression/Practice/Practice2/Regression_Practice_2.ipynb
  git commit -m "feat: implement exercise 2 (model training)"
  ```

---

### Task 4: Implement Exercise 3 (Sales Comparison Plotting)

**Files:**
- Modify: `/home/andv/personal/hm-kpdl/3-Regression/Practice/Practice2/Regression_Practice_2.ipynb` (Cell with ID `ba545a96-a97e-483c-a31c-7fee063f479e`)

- [ ] **Step 1: Check existing cell source**
  Locate cell `ba545a96-a97e-483c-a31c-7fee063f479e` to ensure it only has the exercise description comments.

- [ ] **Step 2: Write minimal implementation**
  Modify `/home/andv/personal/hm-kpdl/3-Regression/Practice/Practice2/Regression_Practice_2.ipynb` by replacing the cell source for `ba545a96-a97e-483c-a31c-7fee063f479e` with:
  ```json
        "source": [
          "###### exercise #####\n",
          "# Yêu cầu: Vẽ biểu đồ đường so sánh y_test và y_pred_test\n",
          "# Gợi ý: sử dụng matplotlib như bài thực hành 1\n",
          "######################\n",
          "plt.figure(figsize=(9, 6))\n",
          "plt.plot(y_test, label='Thực tế (y_test)', color='blue', marker='o')\n",
          "plt.plot(y_pred_test, label='Dự đoán (y_pred_test)', color='red', linestyle='--', marker='x')\n",
          "plt.xlabel('Time index')\n",
          "plt.ylabel('Sales')\n",
          "plt.title('So sánh doanh thu thực tế và doanh thu dự đoán')\n",
          "plt.legend()\n",
          "plt.show()"
        ]
  ```

- [ ] **Step 3: Commit**
  ```bash
  git add 3-Regression/Practice/Practice2/Regression_Practice_2.ipynb
  git commit -m "feat: implement exercise 3 (plotting)"
  ```

---

### Task 5: Implement Exercise 4 (Feature Merging) & Complete Verification

**Files:**
- Modify: `/home/andv/personal/hm-kpdl/3-Regression/Practice/Practice2/Regression_Practice_2.ipynb` (Cell with ID `8640eb1c-a9f5-4f6b-9e68-4732c1d785cf`)

- [ ] **Step 1: Check existing cell source**
  Locate cell `8640eb1c-a9f5-4f6b-9e68-4732c1d785cf` to ensure it only has the exercise description comments.

- [ ] **Step 2: Write minimal implementation**
  Modify `/home/andv/personal/hm-kpdl/3-Regression/Practice/Practice2/Regression_Practice_2.ipynb` by replacing the cell source for `8640eb1c-a9f5-4f6b-9e68-4732c1d785cf` with:
  ```json
        "source": [
          "###### exercise #####\n",
          "# Yêu cầu: Ghép đặc trưng Month_1, ..., Month_12 vào các đặc trưng đang có, kết quả ở dạng numpy array\n",
          "# Gợi ý: sử dụng np.hstack\n",
          "######################\n",
          "X_train = np.hstack((X_train, month_onehot_train.values))"
        ]
  ```

- [ ] **Step 3: Run the verification script to verify it passes**
  Run: `python3 /home/andv/personal/hm-kpdl/docs/superpowers/scratch/verify_notebook.py`
  Expected: PASS with "Notebook executed successfully!" and correct output printed.

- [ ] **Step 4: Commit**
  ```bash
  git add 3-Regression/Practice/Practice2/Regression_Practice_2.ipynb
  git commit -m "feat: implement exercise 4 (feature merging) and verify"
  ```
