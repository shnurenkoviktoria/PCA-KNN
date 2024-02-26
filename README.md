# K-Nearest Neighbors Classifier with PCA

This script demonstrates the use of K-Nearest Neighbors (KNN) classifier with Principal Component Analysis (PCA) for dimensionality reduction on the MNIST dataset.

## Data Preprocessing

The script first loads the MNIST dataset using TensorFlow.keras. It then splits the dataset into training, validation, and test sets. The pixel values are normalized to the range [0, 1] and reshaped to flatten the images. Standard scaling is applied to normalize the feature vectors.

## Dimensionality Reduction with PCA

PCA is applied to reduce the dimensionality of the feature vectors. The script uses 100 principal components for dimensionality reduction.

## K-Nearest Neighbors Classifier

A KNN classifier with 3 neighbors is trained on the PCA-transformed training data.

## Model Evaluation

The trained model is evaluated on the test set, and the accuracy score is calculated using scikit-learn's `metrics.accuracy_score` function.

## Dependencies

- scikit-learn
- TensorFlow (for loading the MNIST dataset)
