# 🔍 Advanced dimensionality reduction visualization using PCA, t-SNE, and UMAP

## 📋 Project Overview

This project implements an advanced dimensionality reduction visualization pipeline using PCA, t-SNE, and UMAP on the Digits dataset from scikit-learn (originally from UCI, commonly used on Kaggle). The pipeline includes hyperparameter tuning, 3D interactive visualizations, and evaluation metrics like trustworthiness and reconstruction error, all implemented in Python for Jupyter or compatible environments.

The pipeline includes:

- Data preprocessing and standardization
- Dimensionality reduction with PCA, t-SNE, and UMAP
- Hyperparameter optimization via grid search
- 3D interactive visualizations using Plotly
- Quantitative evaluation of embeddings

## 🎯 Objectives

- ✅ Reduce high-dimensional Digits dataset to 2D/3D for visualization
- ✅ Optimize t-SNE and UMAP hyperparameters using trustworthiness
- ✅ Evaluate PCA with explained variance and reconstruction error
- ✅ Visualize clusters in 3D with interactive scatter plots
- ✅ Provide a robust framework for comparing dimensionality reduction techniques

## 📊 Dataset Information

- **Source**: Digits dataset (scikit-learn, originally UCI, used in Kaggle tutorials)
- **Size**: 1797 samples, 64 features (8x8 pixel images of handwritten digits), 10 classes
- **Files**:
  - `advanced_dimensionality_reduction.py`: Main Python script
  - `requirements.txt`: Project dependencies
- **Target Output**: 2D/3D embeddings of high-dimensional data, colored by digit class
- **Techniques Used**:
  - PCA for linear dimensionality reduction
  - t-SNE for non-linear, local structure preservation
  - UMAP for non-linear, balanced local/global structure

## 🔧 Technical Implementation

### 📌 Dimensionality Reduction Algorithms

- **PCA**: Principal Component Analysis (linear, 3 components)
- **t-SNE**: t-Distributed Stochastic Neighbor Embedding (non-linear, 3 components)
- **UMAP**: Uniform Manifold Approximation and Projection (non-linear, 3 components)

### 🧹 Data Preprocessing

- Standardize 64 pixel features (mean=0, variance=1) using StandardScaler
- No additional feature engineering required (pixel intensities used directly)

### ⚙️ Modeling

- **PCA**: Fit to capture maximum variance, project to 3D
- **t-SNE**: Grid search over perplexity (5, 30, 50) and learning rate (10, 200, auto)
- **UMAP**: Grid search over n_neighbors (5, 15, 30) and min_dist (0.1, 0.5)
- Project data to 3D for visualization

### 📏 Evaluation Metrics

- **PCA**:
  - Explained variance ratio
  - Reconstruction error (MSE between original and reconstructed data)
- **t-SNE and UMAP**:
  - Trustworthiness (neighborhood preservation metric)

### 📊 Visualizations

- 📉 3D interactive scatter plots (Plotly):
  - PCA (PC1 vs. PC2 vs. PC3)
  - t-SNE (TSNE1 vs. TSNE2 vs. TSNE3)
  - UMAP (UMAP1 vs. UMAP2 vs. UMAP3)
- Points colored by digit class (0-9) using Viridis colorscale
- Dashboard with side-by-side subplots for comparison

## 🚀 Getting Started

### 🧰 Prerequisites

- Python 3.8+
- Jupyter Notebook or VSCode/Jupyter-compatible IDE

### 🛠️ Installation

```bash
git clone https://github.com/zubair-csc/007-Dimensionality_Reduction_Viz_PCA_tSNE_UMAP_Digits.git
cd 007-Dimensionality_Reduction_Viz_PCA_tSNE_UMAP_Digits
pip install -r requirements.txt
```

### 🧪 Usage

1. Launch the Python script or open in Jupyter:
   ```bash
   python advanced_dimensionality_reduction.py
   ```
   or
   ```bash
   jupyter notebook
   ```

2. Run the script to generate:
   - 3D interactive visualizations
   - Evaluation metrics (explained variance, reconstruction error, trustworthiness)

3. Customize (optional):
   - Adjust grid search parameters in the script
   - Modify Plotly visualization settings (e.g., colors, sizes)

## 📈 Results

- **PCA**: Linear embeddings capturing global variance
- **t-SNE**: Non-linear embeddings emphasizing local digit clusters
- **UMAP**: Balanced embeddings with clear class separation
- **Visualizations**: Interactive 3D plots for intuitive exploration
- **Metrics**: Quantitative comparison of embedding quality

## 🙌 Acknowledgments

- scikit-learn for the Digits dataset
- Libraries: pandas, numpy, plotly, scikit-learn, umap-learn, scipy
- Kaggle community for inspiring dimensionality reduction workflows

## ✨ Author

**Zubair** – [zubair-csc](https://github.com/zubair-csc)
