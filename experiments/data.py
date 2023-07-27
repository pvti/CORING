import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


def create_random_centroids(num_centroids, filter_size):
    centroids = np.sqrt(clusters_std) * np.random.rand(num_centroids, *filter_size)
    return centroids


def add_noise_to_centroids(centroids, num_similar_filters, noise_std):
    num_centroids, *filter_size = centroids.shape
    closely_similar_filters = np.zeros(
        [num_centroids * num_similar_filters, *filter_size]
    )
    for i in range(num_centroids):
        for j in range(num_similar_filters):
            similar_filter = centroids[i] + np.sqrt(noise_std) * np.random.randn(
                *filter_size
            )
            closely_similar_filters[i * num_similar_filters + j] = similar_filter
    return closely_similar_filters


filter_size = (3, 3, 64)
num_centroids = 5
clusters_std = 0.5
num_similar_filters = 100
noise_std = 0.1
dataset_file = "synthetic_dataset.npy"

if os.path.exists(dataset_file):
    data = np.load(dataset_file, allow_pickle=True)
    initial_filters = data[0]
    closely_similar_filters = data[1]
else:
    initial_filters = create_random_centroids(num_centroids, filter_size)
    closely_similar_filters = add_noise_to_centroids(
        initial_filters, num_similar_filters, noise_std
    )

    # Save both initial_filters and closely_similar_filters to the file
    np.save(dataset_file, [initial_filters, closely_similar_filters])

# Perform PCA for visualization (with reshaped filters)
reshaped_filters = np.vstack((initial_filters, closely_similar_filters))
pca = PCA(n_components=2)
pca_result = pca.fit_transform(reshaped_filters.reshape(reshaped_filters.shape[0], -1))

# Separate the points of each centroid
centroid_points = pca_result[:num_centroids, :]
similar_points = pca_result[num_centroids:, :]

# Create a colormap with a color for each centroid
colors = plt.cm.tab10(np.linspace(0, 1, num_centroids))

# Plot the points in a 2D plane with different colors for each centroid
plt.figure(figsize=(8, 6))
for i in range(num_centroids):
    # Plot the closely similar filters of the current centroid (cluster)
    plt.scatter(
        similar_points[i * num_similar_filters : (i + 1) * num_similar_filters, 0],
        similar_points[i * num_similar_filters : (i + 1) * num_similar_filters, 1],
        s=50,
        c=[colors[i]],
        label=f"Cluster {i + 1}",
        marker="o",
    )

# Plot the centroids with 'x' markers
plt.scatter(
    centroid_points[:, 0],
    centroid_points[:, 1],
    s=100,
    c="k",
    marker="x",
    label="Centroids",
)

# Set plot labels and title
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("2D PCA Projection. Ground truth")

# Add legend
plt.legend()

# Set aspect ratio to equal and grid on
plt.axis("equal")
plt.grid(True)
plt.show()
