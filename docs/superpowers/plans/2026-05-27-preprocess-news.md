# Preprocess News Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Preprocessing_News.ipynb notebook by implementing Vietnamese stopwords filtering and TF-IDF numerical vectorization.

**Architecture:** Load Vietnamese stopwords, tokenize the stopwords to match word segmentation, apply pyvi.ViTokenizer to segment compound words in the input dataset, and run CountVectorizer and TfidfTransformer to build the TF-IDF representation.

**Tech Stack:** Python 3.11, pyvi, scikit-learn, numpy, nbformat (for verification)

---

### Task 1: Complete Notebook Code Cell
**Files:**
- Modify: `2-Preprocessing/Assignment/News_VNExpress_/Preprocessing_News.ipynb`

- [ ] **Step 1: Implement stopwords loading, Vietnamese tokenization, CountVectorizer, and TfidfTransformer**
  Replace the empty cell block containing `id: Ns9j8Hgrhuof` with the complete implementation.

  ```json
      {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {
          "id": "Ns9j8Hgrhuof"
        },
        "outputs": [],
        "source": [
          "# load dữ liệu các stopwords\n",
          "with open(\"vietnamese-stopwords.txt\", \"r\", encoding=\"utf-8\") as f:\n",
          "    stopwords = [line.strip().replace(\" \", \"_\") for line in f if line.strip()]\n",
          "\n",
          "# Chuyển hoá dữ liệu text về dạng vector TF\n",
          "#     - loại bỏ từ dừng\n",
          "#     - sinh từ điển\n",
          "X_tokenized = [ViTokenizer.tokenize(doc) for doc in data_train.data]\n",
          "\n",
          "module_count_vector = CountVectorizer(stop_words=stopwords)\n",
          "X_counts = module_count_vector.fit_transform(X_tokenized)\n",
          "\n",
          "# Hàm thực hiện chuyển đổi dữ liệu text thành dữ liệu số dạng ma trận\n",
          "# Input: Dữ liệu 2 chiều dạng numpy.array, mảng nhãn id dạng numpy.array\n",
          "tfidf_transformer = TfidfTransformer()\n",
          "data_preprocessed = tfidf_transformer.fit_transform(X_counts)\n",
          "\n",
          "X = data_preprocessed # thuoc tinh\n",
          "Y = data_train.target #nhan\n",
          "\n",
          "print(f\"\\nSố lượng từ trong từ điển: {len(module_count_vector.vocabulary_)}\")\n",
          "print(f\"Kích thước dữ liệu sau khi xử lý: {X.shape}\")\n",
          "print(f\"Kích thước nhãn tương ứng: {Y.shape}\")"
        ]
      }
  ```

### Task 2: Verification
**Files:**
- Create: `2-Preprocessing/Assignment/News_VNExpress_/verify_notebook.py`

- [ ] **Step 1: Write verification script to execute the notebook and verify results**
  Create a script `2-Preprocessing/Assignment/News_VNExpress_/verify_notebook.py` to programmatically run the notebook cells and verify all variables (`X`, `Y`, `module_count_vector`) are computed correctly.

  Code:
  ```python
  import os
  import nbformat
  import sys

  # Change directory to the notebook's parent directory
  os.chdir(os.path.dirname(os.path.abspath(__file__)))

  with open("Preprocessing_News.ipynb", "r", encoding="utf-8") as f:
      nb = nbformat.read(f, as_version=4)

  # Prepare environment for notebook execution
  # We mock %matplotlib inline by overriding it
  global_env = {
      "__file__": "Preprocessing_News.ipynb",
      "__name__": "__main__",
  }

  for i, cell in enumerate(nb.cells):
      if cell.cell_type == "code":
          source = cell.source
          # Filter out magics like %matplotlib inline
          lines = [line for line in source.splitlines() if not line.strip().startswith("%")]
          code = "\n".join(lines)
          try:
              exec(code, global_env)
          except Exception as e:
              print(f"Error in cell {i}:")
              print(code)
              raise e

  # Assertions
  assert "X" in global_env, "X is not defined"
  assert "Y" in global_env, "Y is not defined"
  assert "module_count_vector" in global_env, "module_count_vector is not defined"

  X = global_env["X"]
  Y = global_env["Y"]
  module_count_vector = global_env["module_count_vector"]

  print("\n=== Verification Successful ===")
  print(f"X shape: {X.shape}")
  print(f"Y shape: {Y.shape}")
  print(f"Vocabulary size: {len(module_count_vector.vocabulary_)}")
  ```

- [ ] **Step 2: Run verification script**
  Run: `../.venv/bin/python verify_notebook.py` or `uv run python verify_notebook.py` from `2-Preprocessing/Assignment/News_VNExpress_/` directory.
  Expected: Execution finishes successfully and prints out:
  ```
  Các nhãn và số văn bản tương ứng trong dữ liệu
  ...
  Tổng số văn bản: 1339
  ...
  === Verification Successful ===
  X shape: (1339, ...)
  Y shape: (1339,)
  Vocabulary size: ...
  ```

- [ ] **Step 3: Cleanup verification script**
  Delete the temporary `verify_notebook.py` script so that we leave a clean repository.
