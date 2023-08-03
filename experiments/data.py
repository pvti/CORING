import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


def parse_args():
    parser = argparse.ArgumentParser("Dataset creation")
    parser.add_argument(
        "--data",
        type=str,
        default="dataset_1.npy",
        help="path to dataset",
    )
    parser.add_argument("--centroids", type=int, default=5, help="number of centroids")
    parser.add_argument(
        "--satellites", type=int, default=100, help="number of satellites"
    )
    parser.add_argument(
        "--clusters-std",
        type=float,
        default=1.0,
        help="cluster standard to control the distance among centroids",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.1,
        help="noise standard to control the distance between centroid and its satellites",
    )

    return parser.parse_args()


def create_random_centroids(num_centroids, filter_size, clusters_std):
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


def main():
    args = parse_args()
    filter_size = (3, 3, 64)

    if os.path.exists(args.data):
        data = np.load(args.data, allow_pickle=True)
        initial_filters = data[0]
        closely_similar_filters = data[1]
    else:
        initial_filters = create_random_centroids(
            args.centroids, filter_size, args.clusters_std
        )
        closely_similar_filters = add_noise_to_centroids(
            initial_filters, args.satellites, args.noise_std
        )

        # Save both initial_filters and closely_similar_filters to the file
        np.save(args.data, [initial_filters, closely_similar_filters])

    # Perform PCA for visualization (with reshaped filters)
    reshaped_filters = np.vstack((initial_filters, closely_similar_filters))
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(
        reshaped_filters.reshape(reshaped_filters.shape[0], -1)
    )

    # Separate the points of each centroid
    centroid_points = pca_result[: args.centroids, :]
    similar_points = pca_result[args.centroids :, :]

    # Create a colormap with a color for each centroid
    colors = plt.cm.tab10(np.linspace(0, 1, args.centroids))

    # Plot the points in a 2D plane with different colors for each centroid
    plt.figure(figsize=(8, 6))
    for i in range(args.centroids):
        # Plot the closely similar filters of the current centroid (cluster)
        plt.scatter(
            similar_points[i * args.satellites : (i + 1) * args.satellites, 0],
            similar_points[i * args.satellites : (i + 1) * args.satellites, 1],
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


if __name__ == "__main__":
    main()
