# K-Means Homework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a custom K-Means algorithm using NumPy vectorization within a Jupyter notebook and use it for 2D clustering and image compression.

**Architecture:** We will implement the `CustomKMeans` class containing `fit`, `predict`, and `__init__` methods, relying on NumPy for efficient matrix broadcasting instead of generic Python loops.

**Tech Stack:** Python, Jupyter Notebook, NumPy, Matplotlib, Scikit-learn, Scikit-image

---

### Task 1: Implement `CustomKMeans` Class

**Files:**
- Modify: `4+5-k-mean/Kmeans-Homework/KMeansHomework.ipynb`

- [ ] **Step 1: Write CustomKMeans cell**
  Insert a new Python cell under the "Tự xây dựng giải thuật K-means" section.

```python
import numpy as np

class CustomKMeans:
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        
    def fit(self, X):
        np.random.seed(self.random_state)
        # Randomly choose initial centers from data points
        random_idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.cluster_centers_ = X[random_idx]
        
        for i in range(self.max_iter):
            # Compute distances using broadcasting (N, 1, d) - (1, K, d)
            distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
            
            # Assign labels
            new_labels = np.argmin(distances, axis=1)
            
            # Compute new centers
            new_centers = np.array([X[new_labels == k].mean(axis=0) for k in range(self.n_clusters)])
            
            # Check for convergence
            if np.all(self.cluster_centers_ == new_centers):
                self.labels_ = new_labels
                break
                
            self.cluster_centers_ = new_centers
            self.labels_ = new_labels
            
        return self
        
    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)
```

- [ ] **Step 2: Run the cell**
  Run the newly inserted notebook cell to make the class available in the kernel.

---

### Task 2: Verify Algorithm on 2D Generated Data

**Files:**
- Modify: `4+5-k-mean/Kmeans-Homework/KMeansHomework.ipynb`

- [ ] **Step 1: Create verification cell**
  Insert a new cell under "Kiểm tra giải thuật K-means tự viết cho dữ liệu sinh ngẫu nhiên".

```python
# Initialize both models
sklearn_kmeans = KMeans(n_clusters=3, random_state=170, n_init='auto')
custom_kmeans = CustomKMeans(n_clusters=3, random_state=170)

# Fit models
sklearn_kmeans.fit(X)
custom_kmeans.fit(X)

# Plotting side by side
plt.figure(figsize=(16, 6))

# Sklearn Result
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=sklearn_kmeans.labels_, cmap='viridis')
plt.scatter(sklearn_kmeans.cluster_centers_[:, 0], sklearn_kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title('Sklearn KMeans Clustering')
plt.legend()

# Custom Result
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=custom_kmeans.labels_, cmap='viridis')
plt.scatter(custom_kmeans.cluster_centers_[:, 0], custom_kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title('Custom KMeans Clustering')
plt.legend()

plt.show()
```

- [ ] **Step 2: Run the cell**
  Run the newly inserted notebook cell and verify plots look nearly identical.

---

### Task 3: Apply CustomKMeans to Image Compression

**Files:**
- Modify: `4+5-k-mean/Kmeans-Homework/KMeansHomework.ipynb`

- [ ] **Step 1: Create image compression cell**
  Insert a new cell at the end of the notebook (under "Nén ảnh bằng giải thuật K-means tự viết").
  We use `K=16` colors as typically required for this exercise.

```python
k_colors = 16

# 1. Custom KMeans compression
print(f"Compressing using CustomKMeans with K={k_colors}...")
custom_kmeans_img = CustomKMeans(n_clusters=k_colors, random_state=42)
custom_kmeans_img.fit(data_img)
custom_compressed_img = custom_kmeans_img.cluster_centers_[custom_kmeans_img.labels_]
custom_compressed_img = custom_compressed_img.reshape(img_shape)

# 2. Sklearn KMeans compression
print(f"Compressing using Sklearn KMeans with K={k_colors}...")
sklearn_kmeans_img = KMeans(n_clusters=k_colors, random_state=42, n_init='auto')
sklearn_kmeans_img.fit(data_img)
sklearn_compressed_img = sklearn_kmeans_img.cluster_centers_[sklearn_kmeans_img.labels_]
sklearn_compressed_img = sklearn_compressed_img.reshape(img_shape)

# 3. Plotting Original, Sklearn and Custom
plt.figure(figsize=(20, 6))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sklearn_compressed_img)
plt.title(f'Sklearn KMeans (K={k_colors})')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(custom_compressed_img)
plt.title(f'Custom KMeans (K={k_colors})')
plt.axis('off')

plt.show()
```

- [ ] **Step 2: Run the cell**
  Run the compression cell and verify it outputs the three corresponding images properly.

- [ ] **Step 3: Commit the changes**
  Commit the notebook changes indicating completion of the homework.
```bash
git add docs/superpowers/specs/2026-05-31-kmeans-homework-implementation-design.md
git add docs/superpowers/plans/2026-05-31-kmeans-homework.md
git add 4+5-k-mean/Kmeans-Homework/KMeansHomework.ipynb
git commit -m "feat: complete custom KMeans algorithm and image compression homework"
```
