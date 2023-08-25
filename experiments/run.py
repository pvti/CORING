import argparse
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import silhouette_score
from kmeans import custom_kmeans, compute_inter_distance, compute_intra_distance
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
    parser.add_argument("--rank", type=int, default=1, help="decomposition rank")
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


def run(seed=0):
    # Apply custom K-means on the dataset
    centroids, labels, inertia_list, iter = custom_kmeans(
        data_combined,
        num_clusters,
        dist=args.distance,
        decomposer=args.method,
        rank=args.rank,
        seed=seed,
    )

    inertia = min(inertia_list)
    try:
        silhouette = silhouette_score(data_combined_2d, labels)
        intra_distance = compute_intra_distance(centroids, data_combined, labels)
        inter_distance = compute_inter_distance(centroids)

    except Exception:
        silhouette = 0

    return inertia, silhouette, iter, intra_distance, inter_distance


def compute_std(x):
    min = np.min(x)
    avg = np.mean(x)
    max = np.max(x)

    return min, avg, max


def main():
    name = f"{args.data}_{args.method}_{args.distance}_{args.rank}"
    wandb.init(name=name, project=f"CORING_CustomKmeans", config=vars(args))

    inertia_values = []
    silhouette_values = []
    intra_distances = []
    inter_distances = []

    for i in tqdm(range(args.runs)):
        inertia, silhouette, iter, intra_distance, inter_distance = run(seed=i)
        wandb.log(
            {
                "inertia": inertia,
                "silhouette": silhouette,
                "intra_distance": intra_distance,
                "inter_distance": inter_distance,
                "iteration to converged": iter,
            }
        )
        inertia_values.append(inertia)
        silhouette_values.append(silhouette)
        intra_distances.append(intra_distance)
        inter_distances.append(inter_distance)

    inertia_min, inertia_avg, inertia_max = compute_std(inertia_values)
    silhouette_min, silhouette_avg, silhouette_max = compute_std(silhouette_values)
    intra_min, intra_avg, intra_max = compute_std(intra_distances)
    inter_min, inter_avg, inter_max = compute_std(inter_distances)

    print("inertia_min:", inertia_min)
    print("inertia_avg:", inertia_avg)
    print("inertia_max:", inertia_max)
    print("silhouette_min:", silhouette_min)
    print("silhouette_avg:", silhouette_avg)
    print("silhouette_max:", silhouette_max)
    print("intra_min", intra_min)
    print("intra_avg", intra_avg)
    print("intra_max", intra_max)
    print("inter_min", inter_min)
    print("inter_avg", inter_avg)
    print("inter_max", inter_max)


if __name__ == "__main__":
    main()
