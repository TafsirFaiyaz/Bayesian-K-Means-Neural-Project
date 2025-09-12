# # Bayesian Deep K-Means with Normalizing Flows (BDKM-NF)
# 
# This notebook implements a novel clustering approach that combines:
# - Bayesian deep learning principles
# - Differentiable K-means clustering
# - Contrastive learning
# - Uncertainty quantification
# 
# ## Key Features:
# - **Differentiable K-means**: Soft cluster assignments with temperature annealing
# - **Uncertainty Analysis**: Monte Carlo sampling for robustness
# - **Contrastive Learning**: NT-Xent loss for better representations
# - **Silhouette-aware Loss**: Encourages well-separated clusters
# - **UMAP/PCA Preprocessing**: Dimensionality reduction before final clustering