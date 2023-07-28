import argparse
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import silhouette_score
from kmeans import custom_kmeans, euclidean_distance
import wandb


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
    parser.add_argument("--runs", type=int, default=100, help="number of runs")

    return parser.parse_args()


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

data_combined_2d = data_combined.reshape(data_combined.shape[0], -1)


def run():
    # Apply custom K-means on the dataset
    centroids, labels, inertia_list, iter = custom_kmeans(
        data_combined, num_clusters, distance_func=euclidean_distance
    )

    inertia = min(inertia_list)
    try:
        silhouette = silhouette_score(data_combined_2d, labels)
    except Exception:
        silhouette = 0

    return inertia, silhouette, iter


def main():
    name = f"{args.method}_{args.distance}"
    wandb.init(name=name, project=f"CORING_CustomKmeans", config=vars(args))

    inertia_values = []
    silhouette_values = []

    for i in tqdm(range(args.runs)):
        inertia, silhouette, iter = run()
        wandb.log(
            {
                "inertia": inertia,
                "silhouette": silhouette,
                "iteration to converged": iter,
            }
        )
        inertia_values.append(inertia)
        silhouette_values.append(silhouette)

    inertia_avg = np.mean(inertia_values)
    silhouette_avg = np.mean(silhouette_values)
    inertia_min = np.min(inertia_values)
    inertia_max = np.max(inertia_values)
    silhouette_min = np.min(silhouette_values)
    silhouette_max = np.max(silhouette_values)

    print("inertia_avg:", inertia_avg)
    print("silhouette_avg:", silhouette_avg)
    print("inertia_min:", inertia_min)
    print("inertia_max:", inertia_max)
    print("silhouette_min:", silhouette_min)
    print("silhouette_max:", silhouette_max)


if __name__ == "__main__":
    main()
