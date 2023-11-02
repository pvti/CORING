import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from utils import svd, tensor_decompose, VBD


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
        default="tensor",
        choices=("tensor", "matrix"),
        help="decomposition method",
    )
    parser.add_argument(
        "--distance",
        type=str,
        default="euclidean",
        choices=("euclidean", "vbd", "cosine"),
        help="distance metric",
    )
    parser.add_argument("--rank", type=int, default=1, help="decomposition rank")
    parser.add_argument("--seed", type=int, default=0, help="seed for random")

    return parser.parse_args()


def decompose(data, method="tensor", rank=1):
    representations = []
    for d in data:
        if method == "matrix":
            factors = svd(d, rank=rank)
        elif method == "tensor":
            factors = tensor_decompose(d, rank=rank)
        representations.append(factors)

    return np.vstack(representations)


def compute_distance(x, y, dist="euclidean"):
    sum = 0.0
    num_representation = len(x)
    for i in range(num_representation):
        if dist == "euclidean":
            sum += np.linalg.norm(x[i] - y[i])
        elif dist == "vbd":
            sum += VBD(x[i], y[i])

    distance = sum.item() / num_representation

    return distance


def custom_kmeans(
    data,
    data_processed,
    method="tensor",
    rank=1,
    num_clusters=5,
    dist="euclidean",
    max_iters=100,
    tolerance=1e-12,
    seed=0,
):
    random_state = np.random.RandomState(seed)
    # Randomly initialize the centroids from the data points
    centroids_idx = random_state.choice(data.shape[0], num_clusters, replace=False)
    centroids = data[centroids_idx]
    # centroids = [data_processed[i] for i in centroids_idx]
    centroids_processed = data_processed[centroids_idx]

    inertia_list = []

    for iter in range(max_iters):
        # Assign each data point to the nearest centroid
        labels = np.argmin(
            np.array(
                [
                    [
                        compute_distance(
                            d,
                            c,
                            dist=dist,
                        )
                        for c in centroids_processed
                    ]
                    for d in data_processed
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
        centroids_processed = decompose(new_centroids, method=method, rank=rank)

    return centroids, labels, inertia_list, iter


def compute_intra_distance(centroids, data, labels):
    intra_distance = 0.0
    for i in range(len(centroids)):
        centroid = centroids[i]
        satellite_indices = np.where(labels == i)[0]
        satellites = data[satellite_indices]
        for satellite in satellites:
            intra_distance += np.linalg.norm(satellite - centroid)
    return intra_distance


def compute_inter_distance(centroids):
    inter_distance = 0.0
    num_centroids = len(centroids)
    for i in range(num_centroids):
        for j in range(i + 1, num_centroids):
            inter_distance += np.linalg.norm(centroids[i] - centroids[j])
    return inter_distance


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
    data_processed = decompose(data_combined, method=args.method, rank=args.rank)

    # Apply custom K-means on the dataset
    centroids, labels, inertia_list, iteration = custom_kmeans(
        data=data_combined,
        data_processed=data_processed,
        method=args.method,
        rank=args.rank,
        num_clusters=num_clusters,
        dist=args.distance,
        seed=args.seed,
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

    # Calculate ARI
    num_similar_filters = closely_similar_filters.shape[0] // num_clusters
    ground_truth = np.arange(num_clusters)
    temp = np.repeat(ground_truth, num_similar_filters)
    ground_truth = np.concatenate([ground_truth, temp])
    ari = adjusted_rand_score(ground_truth, labels)
    print("ARI:", ari)

    # Compute Intra Distance
    intra_distance = compute_intra_distance(centroids, data_combined, labels)
    print("Intra Distance:", intra_distance)

    # Compute Inter Distance
    inter_distance = compute_inter_distance(centroids)
    print("Inter Distance:", inter_distance)

    # Visualize the clustered data using PCA in 2-D space
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_combined.reshape(data_combined.shape[0], -1))

    # Separate the points of each centroid
    centroid_points = pca.transform(centroids.reshape(centroids.shape[0], -1))

    # Create a colormap with a color for each centroid
    colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))

    # Plot the points in a 2D plane with different colors for each cluster
    plt.figure(figsize=(8, 6), dpi=300)
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
    # plt.xlabel("Principal Component 1")
    # plt.ylabel("Principal Component 2")
    # plt.title(
    #     f"data = {args.data}; method = {args.method}; ARI = {ari}"
    # )

    # Add legend
    # plt.legend(prop = { "size": 13 })
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    # Set aspect ratio to equal and grid on
    # plt.axis("equal")
    plt.grid(True)
    plt.show()
