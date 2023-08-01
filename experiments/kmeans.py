import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
from sklearn.metrics import silhouette_score

import sys

sys.path.append("..")
from decompose import decompose


def parse_args():
    parser = argparse.ArgumentParser("Custom metric for K-means clustering")
    parser.add_argument(
        "--data",
        type=str,
        default="dataset_1.npy",
        help="path to dataset",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="hosvd",
        choices=("hosvd", "svd"),
        help="decomposition method",
    )
    parser.add_argument(
        "--distance",
        type=str,
        default="euclidean",
        choices=("euclidean", "vbd", "cosine"),
        help="distance metric",
    )

    return parser.parse_args()


def compute_distance(x, y, dist="euclidean", decomposer="hosvd"):
    ux = decompose(torch.tensor(x), decomposer=decomposer)
    uy = decompose(torch.tensor(y), decomposer=decomposer)

    if decomposer == "svd":
        if dist == "euclidean":
            distance = torch.dist(ux[0], uy[0])
        elif dist == "vbd":
            distance = torch.var(ux[0] - uy[0]) / (torch.var(ux[0]) + torch.var(uy[0]))

    elif decomposer == "hosvd":
        sum = 0.0
        for i in range(3):
            if dist == "euclidean":
                sum += torch.dist(ux[i], uy[i])
            elif dist == "vbd":
                sum += torch.var(ux[i] - uy[i]) / (torch.var(ux[i]) + torch.var(uy[i]))

        distance = sum.item() / 3

    return distance


def custom_kmeans(
    data,
    num_clusters,
    dist="euclidean",
    decomposer="hosvd",
    max_iters=100,
    tolerance=1e-12,
):
    # Randomly initialize the centroids from the data points
    centroids_idx = np.random.choice(data.shape[0], num_clusters, replace=False)
    centroids = data[centroids_idx]

    inertia_list = []

    for iter in range(max_iters):
        # Assign each data point to the nearest centroid
        labels = np.argmin(
            np.array(
                [
                    [
                        compute_distance(d, c, dist=dist, decomposer=decomposer)
                        for c in centroids
                    ]
                    for d in data
                ]
            ),
            axis=1,
        )

        # Update the centroids as the mean of the data points in each cluster
        new_centroids = np.array(
            [data[labels == i].mean(axis=0) for i in range(num_clusters)]
        )

        # Compute inertia
        inertia = np.sum(
            [
                np.linalg.norm(data[labels == i] - new_centroids[i])
                for i in range(num_clusters)
            ]
        )
        inertia_list.append(inertia)

        # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tolerance:
            break

        centroids = new_centroids

    return centroids, labels, inertia_list, iter


if __name__ == "__main__":
    args = parse_args()
    # Load the synthetic dataset from the file (saved in .npy format)
    dataset_file = args.data
    data = np.load(dataset_file, allow_pickle=True)
    initial_filters = data[0]
    closely_similar_filters = data[1]

    # Set the number of clusters (you can adjust this value as per your requirement)
    num_clusters = 5

    # Combine both initial_filters and closely_similar_filters
    data_combined = np.vstack((initial_filters, closely_similar_filters))

    # Apply custom K-means on the dataset
    centroids, labels, inertia_list, iteration = custom_kmeans(
        data_combined, num_clusters, dist=args.distance, decomposer=args.method
    )

    # Print the inertia list for each iteration
    print("Inertia List:", inertia_list)

    # Calculate silhouette score
    # Reshape data_combined to 2-dimensional for silhouette score calculation
    data_combined_2d = data_combined.reshape(data_combined.shape[0], -1)
    # Calculate silhouette score
    try:
        silhouette = silhouette_score(data_combined_2d, labels)
    except Exception:
        silhouette = 0
    print("silhouette_avg:", silhouette)

    # Visualize the clustered data using PCA in 2-D space
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_combined.reshape(data_combined.shape[0], -1))

    # Separate the points of each centroid
    centroid_points = pca.transform(centroids.reshape(centroids.shape[0], -1))

    # Create a colormap with a color for each centroid
    colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))

    # Plot the points in a 2D plane with different colors for each cluster
    plt.figure(figsize=(8, 6))
    for i in range(num_clusters):
        plt.scatter(
            pca_result[labels == i, 0],
            pca_result[labels == i, 1],
            s=50,
            c=[colors[i]],
            label=f"Cluster {i + 1}",
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
    plt.title(f"{args.method} {args.distance}")

    # Add legend
    plt.legend()

    # Set aspect ratio to equal and grid on
    plt.axis("equal")
    plt.grid(True)
    plt.show()
