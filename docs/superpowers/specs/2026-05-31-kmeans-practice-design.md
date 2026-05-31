# KMeans Practice - Exercises Solution Design

## Overview
This document specifies the design and approach for solving the four exercises in the `KMeans.ipynb` practice notebook.

## General Approach
Instead of writing repetitive code for each condition, we will use loops (e.g., `for k in [3, 4, 5]`) and Matplotlib subplots. This ensures the notebook remains clean, professional, and easy to read, while making visual comparisons side-by-side much easier.

## Detailed Specifications

### 1. Exercise 1: Simulate subset of data with 2 clusters, test with k=3, 4, 5
- **Data Generation**: Use `make_blobs` with `n_samples=100`, `centers=2`, and `random_state=170` to create a dataset `X_ex1` natural consisting of 2 clusters.
- **Modeling & Visualization**: 
  - Iterate through `k_values = [3, 4, 5]`.
  - For each `k`, instantiate and fit a `KMeans` model.
  - Predict the labels and get the cluster centers.
  - Plot the results using a `1x3` subplot grid (`plt.subplots(1, 3, ...)`).
  - Clearly mark the centers in red and color points based on the predicted labels.

### 2. Exercise 2: Image Compression with k < 5
- **Configuration**: Choose `k = 4` (which satisfies $< 5$).
- **Modeling**: Instantiate and train a `KMeans` model with `n_clusters=4` on the existing `data_img`.
- **Reconstruction**: Replace the color of each pixel with the RGB value of its assigned cluster center.
- **Visualization & Evaluation**: 
  - Reshape the processed array back to the original image dimensions `img_shape`.
  - Save the new image as `img128_k4.png`.
  - Display the original and compressed images side-by-side.
  - Print out the file sizes using `os.path.getsize` to demonstrate the compression ratio.

### 3. Bonus Exercise 1: Elbow Method (Inertia) for Blob Data
- **Calculation**: Create an empty list `inertias = []`.
- **Iteration**: Loop `k` from 1 to 10, fitting a `KMeans` model on the dataset `X` (the original 3-cluster data).
- **Metric**: Append `model.inertia_` to the list.
- **Visualization**: Plot a line graph of `inertias` vs. Number of Clusters ($k$) using a marker like `bx-`.
- **Analysis**: The graph should clearly show a "knee/elbow" at $k=3$, which is the optimal number of clusters for this dataset.

### 4. Bonus Exercise 2: Elbow Method (Inertia) for Image Data
- **Calculation**: Create an empty list `inertias_img = []`.
- **Iteration**: Loop `k` from 1 to 10, fitting a `KMeans` model on `data_img`.
- **Visualization**: Plot a line graph to visualize the inertia reduction as the number of colors increases.
- **Analysis**: This will show rapid drops in initial clusters and dimishing returns as $k$ gets larger, helping visualize the trade-off between image quality and size.
