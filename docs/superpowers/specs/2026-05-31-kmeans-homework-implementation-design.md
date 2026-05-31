# Custom K-Means Implementation Design

## Overview
Implementing a custom K-Means clustering algorithm using NumPy vectorization. The goal is to match the core functionality of `sklearn.cluster.KMeans` and apply it to both a randomly generated 2D dataset and an image compression task.

## Architecture: `CustomKMeans` Class
The class will expose an API similar to scikit-learn for ease of use.

### Methods
- `__init__(self, n_clusters=8, max_iter=300, random_state=None)`: 
  - Initializes the algorithm's hyperparameters.
- `fit(self, X)`: 
  - **Initialization**: Randomly select `n_clusters` points directly from the input array `X` to serve as initial centroids. It will use `np.random.RandomState(random_state)` for reproducible results.
  - **Iteration loop** (max `max_iter` times):
    - *Distance Calculation*: Use NumPy broadcasting to compute distances. 
      `distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)` 
      This efficiently computes the distance from every point to every centroid without Python `for` loops.
    - *Assignment*: Assign points to the closest centroid using `labels = np.argmin(distances, axis=1)`.
    - *Update*: Compute the mean of the points assigned to each cluster to find the new centroids.
    - *Convergence Check*: Break the loop early if the new centroids are exactly equivalent to the old centroids.
  - **State**: Save `self.cluster_centers_` and `self.labels_` after the loop.
- `predict(self, X)`: 
  - Calculates the distance from `X` to `self.cluster_centers_` and returns the index of the closest centroid.

## Workflow

### 1. Algorithm Verification (2D Data)
- Generate a 2D dataset (100 samples, 3 centers) using `make_blobs`.
- Initialize both `sklearn.cluster.KMeans` and `CustomKMeans`.
- Fit both models on the generated data.
- Plot the results side-by-side using `matplotlib` to visually confirm that the custom implementation successfully clusters the points and identifies the correct centroids.

### 2. Image Compression Application
- **Data Preparation**: Read `bird_small.png`, normalize pixel values to a `[0, 1]` range, and reshape the 3D local matrix `(height, width, RGB)` into a 2D matrix `(N, 3)` where `N = height * width`.
- **Clustering**: Apply `CustomKMeans(n_clusters=16)` to the pixel data. (Finding 16 representative colors).
- **Compression**: Replace each pixel in the 2D matrix with the RGB value of its assigned centroid.
- **Reconstruction**: Reshape the compressed 2D matrix back to its original 3D dimensions.
- **Comparison**: Plot the Original Image, Sklearn Compressed Image, and Custom Compressed Image side-by-side using `matplotlib` for final evaluation.