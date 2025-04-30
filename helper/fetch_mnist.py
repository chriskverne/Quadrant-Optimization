import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
import pandas as pd

def fetch_mnist(digits, num_images_per_digit):
    # Fetch MNIST dataset
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data.to_numpy()
    y = mnist.target.astype(int).to_numpy()
    
    # Initialize arrays to store selected images and their labels
    selected_images = []
    selected_labels = []
    
    # For each requested digit
    for digit in digits:
        # Find indices where target equals the current digit
        indices = np.where(y == digit)[0]
        
        # Select the specified number of images
        selected_count = min(num_images_per_digit, len(indices))
        digit_indices = indices[:selected_count]
        
        # Add the selected images and their labels to our result
        selected_images.append(X[digit_indices])
        selected_labels.append(np.full(selected_count, digit))
    
    # Combine all selected images and labels
    X_selected = np.vstack(selected_images)
    y_selected = np.concatenate(selected_labels)
    
    # Create random permutation for shuffling
    permutation = np.random.permutation(len(X_selected))
    
    # Return shuffled images and their corresponding labels
    return X_selected[permutation], y_selected[permutation]

def preprocess_image(x, n_components):
    scaler = StandardScaler()
    pca = PCA(n_components=n_components)

    # reduce dimension
    x_scaled = scaler.fit_transform(x)
    x_pca = pca.fit_transform(x_scaled)

    # Normalize to [0, 2π] for angle encoding
    x_pca_normalized = 2.0 * np.pi * (x_pca - x_pca.min(axis=0)) / (x_pca.max(axis=0) - x_pca.min(axis=0))

    return x_pca_normalized

# # 5k train, 10k test
# X, y = fetch_mnist([0,1,2,3], 10000)
# df = pd.DataFrame(X)
# df['label'] = y
# df.to_csv('../data/test_four_digit.csv', index=False)