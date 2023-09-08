import argparse
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import adjusted_rand_score
from kmeans import custom_kmeans, decompose
import wandb
import random


def parse_args():
    parser = argparse.ArgumentParser("Custom metric for K-means clustering")
    parser.add_argument(
        "--runs",
        type=int,
        default=1000,
        help="number of runs",
    )
    parser.add_argument("--centroids", type=int, default=5, help="number of centroids")
    parser.add_argument(
        "--satellites", type=int, default=100, help="number of satellites"
    )
    parser.add_argument(
        "--clusters_std_min",
        type=float,
        default=1.0,
        help="min cluster standard to control the distance among centroids",
    )
    parser.add_argument(
        "--clusters_std_max",
        type=float,
        default=2.0,
        help="max cluster standard to control the distance among centroids",
    )
    parser.add_argument(
        "--noise_std_min",
        type=float,
        default=0.1,
        help="min noise standard to control the distance between centroid and its satellites",
    )
    parser.add_argument(
        "--noise_std_max",
        type=float,
        default=0.5,
        help="max noise standard to control the distance between centroid and its satellites",
    )
    parser.add_argument(
        "--distance",
        type=str,
        default="euclidean",
        choices=("euclidean", "vbd", "cosine"),
        help="distance metric",
    )
    parser.add_argument("--rank", type=int, default=1, help="decomposition rank")
    parser.add_argument(
        "--initialization", type=int, default=100, help="number of initialization"
    )

    return parser.parse_args()


args = parse_args()
filter_size = (3, 3, 64)
ground_truth = np.arange(args.centroids)
temp = np.repeat(ground_truth, args.satellites)
ground_truth = np.concatenate([ground_truth, temp])


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


def run(data_combined, data_processed, method, rank):
    inertia_min = float("inf")
    for i in range(args.initialization):
        # Apply custom K-means on the dataset
        _, labels, inertia_list, _ = custom_kmeans(
            data=data_combined,
            data_processed=data_processed,
            method=method,
            rank=rank,
            num_clusters=args.centroids,
            dist=args.distance,
            seed=i,
        )

        inertia = inertia_list[-1]
        if inertia < inertia_min:
            inertia_min = inertia
            ARI = adjusted_rand_score(ground_truth, labels)

    return ARI, inertia_min


def compute_std(x):
    min = np.min(x)
    avg = np.mean(x)
    max = np.max(x)

    return min, avg, max


def topk_positions(x, y, k=10):
    # Calculate mean values
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate total_diff for each position
    total_diff = np.abs(x - x_mean) + np.abs(y - y_mean)

    # Get indices that would sort the total_diff array
    sorted_indices = np.argsort(total_diff)

    # Get the top k positions with the smallest total_diff
    top_k_positions = sorted_indices[:k]

    return top_k_positions


def main():
    name = f"distance = {args.distance} rank = {args.rank} clusters-std = [{args.clusters_std_min} {args.clusters_std_max}] noise-std = [{args.noise_std_min} {args.noise_std_max}]"
    wandb.init(name=name, project=f"CORING_CustomKmeans", config=vars(args))

    ARIs_tensor = []
    inertias_tensor = []
    ARIs_matrix = []
    inertias_matrix = []

    for i in tqdm(range(args.runs)):
        # Create dataset
        clusters_std = random.uniform(args.clusters_std_min, args.clusters_std_max)
        noise_std = random.uniform(args.noise_std_min, args.noise_std_max)
        print(
            f"Creating dataset {i} with (clusters_std, noise_std)=({clusters_std},{noise_std})"
        )
        initial_filters = create_random_centroids(
            args.centroids, filter_size, clusters_std
        )
        closely_similar_filters = add_noise_to_centroids(
            initial_filters, args.satellites, noise_std
        )
        np.save(f"{i}.npy", [initial_filters, closely_similar_filters])

        # Combine both initial_filters and closely_similar_filters
        data_combined = np.vstack((initial_filters, closely_similar_filters))
        data_processed_tensor = decompose(
            data_combined, method="tensor", rank=args.rank
        )
        data_processed_matrix = decompose(
            data_combined, method="matrix", rank=args.rank
        )

        # Apply K-means
        try:
            ARI_tensor, inertia_tensor = run(
                data_combined=data_combined,
                data_processed=data_processed_tensor,
                method="tensor",
                rank=args.rank,
            )
            ARI_matrix, inertia_matrix = run(
                data_combined=data_combined,
                data_processed=data_processed_matrix,
                method="matrix",
                rank=args.rank,
            )

            wandb.log(
                {
                    "ARI_tensor": ARI_tensor,
                    "inertia_tensor": inertia_tensor,
                    "ARI_matrix": ARI_matrix,
                    "inertia_matrix": inertia_matrix,
                }
            )

            ARIs_tensor.append(ARI_tensor)
            inertias_tensor.append(inertia_tensor)
            ARIs_matrix.append(ARI_matrix)
            inertias_matrix.append(inertia_matrix)

        except Exception as error:
            print(error)

    _, ARI_mean_tensor, _ = compute_std(ARIs_tensor)
    _, ARI_mean_matrix, _ = compute_std(ARIs_matrix)

    print("ARI_mean_tensor", ARI_mean_tensor)
    print("ARI_mean_matrix", ARI_mean_matrix)
    ARIs_tensor = np.array(ARIs_tensor)
    ARIs_matrix = np.array(ARIs_matrix)
    top_positions = topk_positions(ARIs_tensor, ARIs_matrix)
    for i, position in enumerate(top_positions):
        print(
            f"Position {i + 1}: ARIs_tensor[{position}] = {ARIs_tensor[position]}, ARIs_matrix[{position}] = {ARIs_matrix[position]}"
        )


if __name__ == "__main__":
    main()
