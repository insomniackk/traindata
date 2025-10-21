# traindata

Overview
Deep learning pipeline for endometrial ultrasound images:
- Train a ResNet50-based binary classifier (Benign vs Malignant).
- Extract feature embeddings from the penultimate layer.
- Visualize embeddings with UMAP (2D/3D).
- Compare new images to the clean dataset and flag outliers.

Features
- Dataset split: train/validation
- ResNet50 modified for grayscale input
- Training with AdamW + Cosine Annealing LR
- Validation accuracy reporting
- Embedding extraction & UMAP visualization
- Distance-based outlier detection for new images

Outputs
- resnet_ultrasound.pt – trained model
- UMAP plots – 2D/3D visualizations
- distances.txt – distances of new images to cluster center
- Histogram of distances

Dependencies
pip install torch torchvision tqdm umap-learn matplotlib seaborn scikit-learn pillow

Purpose
- Train a reliable classifier.
- Extract embeddings for analysis.
- Quickly detect noisy or outlier images in new datasets.
- Visualize dataset structure and model performance.

