# KMeans Practice Exercises Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the solutions for the 4 exercises in the `KMeans.ipynb` Jupyter notebook.

**Architecture:** We will edit specific cells in the Jupyter notebook `4+5-k-mean/Kmeans-Practice/KMeans.ipynb` using `edit_notebook_file`. We will then use `run_notebook_cell` to verify the execution.

**Tech Stack:** Python, Jupyter, Scikit-Learn, Matplotlib, Numpy.

---

### Task 1: Exercise 1 (Simulate 2-cluster data and run k=3,4,5)

**Files:**
- Modify: `4+5-k-mean/Kmeans-Practice/KMeans.ipynb`

- [ ] **Step 1: Write the implementation**
Use `edit_notebook_file` to replace the content of the cell that starts with `# Sinh dữ liệu tương ứng với giả thiết có 2 cụm` with the following code. (Note: use `edit_notebook_file` or `replace_string_in_file` targeting the contents exactly).

```python
# Sinh dữ liệu tương ứng với giả thiết có 2 cụm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Tạo dữ liệu ngẫu nhiên nhưng chỉ có 2 tâm (2 clusters)
n_samples_ex1 = 100
random_state_ex1 = 170
X_ex1, y_ex1 = make_blobs(
    n_samples=n_samples_ex1,
    random_state=random_state_ex1,
    centers=2,  # Chỉ có 2 cụm tự nhiên
    cluster_std=0.6,
)

# Xây dựng mô hình với 3, 4, 5 cụm và trực quan hóa kết quả phân cụm tương ứng.
k_values = [3, 4, 5]
plt.figure(figsize=(18, 5))

for i, k in enumerate(k_values):
    kmeans_ex1 = KMeans(n_clusters=k, random_state=random_state_ex1)
    kmeans_ex1.fit(X_ex1)
    y_pred_ex1 = kmeans_ex1.predict(X_ex1)
    centers_ex1 = kmeans_ex1.cluster_centers_
    
    plt.subplot(1, 3, i + 1)
    plt.scatter(X_ex1[:, 0], X_ex1[:, 1], c=y_pred_ex1)
    plt.scatter(centers_ex1[:, 0], centers_ex1[:, 1], c="red", marker='X', s=200, label='Mức độ nén (Centers)')
    plt.title(f"Kết quả phân cụm với k = {k}")

plt.show()
```

- [ ] **Step 2: Run the cell to verify it works**
Use `run_notebook_cell` on the modified cell ID to verify the execution.

- [ ] **Step 3: Commit**
```bash
git add "4+5-k-mean/Kmeans-Practice/KMeans.ipynb"
git commit -m "feat: complete exercise 1 with subplots for k=3,4,5"
```


### Task 2: Exercise 2 (Image compression with k < 5)

**Files:**
- Modify: `4+5-k-mean/Kmeans-Practice/KMeans.ipynb`

- [ ] **Step 1: Write the implementation**
Use `edit_notebook_file` to replace the content of the cell containing exactly `# code` after `### Bài tập 2` with the following code.

```python
# Chọn số lượng màu k = 4 (< 5)
n_color_2 = 4
k_mean_model_2 = KMeans(n_clusters=n_color_2, random_state=42)

# Huấn luyện mô hình
k_mean_model_2.fit(data_img)

# Tạo lại ảnh
img128_k4 = k_mean_model_2.cluster_centers_[k_mean_model_2.labels_]
img128_k4 = np.reshape(img128_k4, img_shape)

# Lưu ảnh mới
image.imsave("img128_k4.png", img128_k4)

# Hiển thị và so sánh dung lượng ảnh
import os
print("Size of image compressed (k=4): " + str(os.path.getsize("img128_k4.png")) + " Bytes")
print("Size of original image: " + str(os.path.getsize("bird_small.png")) + " Bytes")

# Trực quan hóa
display(Image("img128_k4.png", width=250, unconfined=True))
display(Image(path_img, width=250, unconfined=True))
```

- [ ] **Step 2: Run the cell to verify it works**
Use `run_notebook_cell` on the modified cell ID. This might take a few seconds due to the image size.

- [ ] **Step 3: Commit**
```bash
git add "4+5-k-mean/Kmeans-Practice/KMeans.ipynb"
git commit -m "feat: complete exercise 2 - image compression with k=4"
```


### Task 3: Bonus Exercise 1 (Elbow graph for blob data)

**Files:**
- Modify: `4+5-k-mean/Kmeans-Practice/KMeans.ipynb`

- [ ] **Step 1: Write the implementation**
Locate the cell with `# Viết code tính inertia_ cho mô hình của dữ liệu điểm ban đầu...` and replace it with:

```python
### Bài tập ###
# Viết code tính inertia_ cho mô hình của dữ liệu điểm ban đầu, số lượng cụm từ 1 đến 10
inertias = []
K = range(1, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km = km.fit(X)
    inertias.append(km.inertia_)

# Vẽ đồ thị để quan sát sự giảm của inertia và chọn số lượng cụm phù hợp
plt.figure(figsize=(8, 5))
plt.plot(K, inertias, 'bx-')
plt.xlabel('Số lượng cụm (k)')
plt.ylabel('Inertia')
plt.title('Phương pháp Elbow xác định k tối ưu (Dữ liệu ban đầu)')
plt.xticks(K)
plt.axvline(x=3, color='r', linestyle='--', label='Điểm Elbow (k=3)')
plt.legend()
plt.show()
```

- [ ] **Step 2: Run the cell to verify it works**
Use `run_notebook_cell` on the newly edited cell.

- [ ] **Step 3: Commit**
```bash
git add "4+5-k-mean/Kmeans-Practice/KMeans.ipynb"
git commit -m "feat: complete bonus exercise 1 - elbow method for raw blobs data"
```


### Task 4: Bonus Exercise 2 (Elbow graph for image data)

**Files:**
- Modify: `4+5-k-mean/Kmeans-Practice/KMeans.ipynb`

- [ ] **Step 1: Write the implementation**
Locate the cell with `# Viết code tính inertia_ cho mô hình của dữ liệu hình ảnh...` and replace it with:

```python
### Bài tập ###
# Viết code tính inertia_ cho mô hình của dữ liệu hình ảnh, số lượng cụm từ 1 đến 10
inertias_img = []
K_img = range(1, 11)
for k in K_img:
    km_img = KMeans(n_clusters=k, random_state=42)
    km_img = km_img.fit(data_img)
    inertias_img.append(km_img.inertia_)

# Vẽ đồ thị để quan sát sự giảm của inertia và chọn số lượng cụm phù hợp
plt.figure(figsize=(8, 5))
plt.plot(K_img, inertias_img, 'gx-')
plt.xlabel('Số lượng cụm (k)')
plt.ylabel('Inertia')
plt.title('Phương pháp Elbow xác định k tối ưu (Dữ liệu ảnh)')
plt.xticks(K_img)
plt.show()
```

- [ ] **Step 2: Run the cell to verify it works**
Use `run_notebook_cell` on the edited cell.

- [ ] **Step 3: Commit**
```bash
git add "4+5-k-mean/Kmeans-Practice/KMeans.ipynb"
git commit -m "feat: complete bonus exercise 2 - elbow method for image data"
```