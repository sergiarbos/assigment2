import argparse
import math
from random import Random
from utils import read_csv


class KMeans:
    def __init__(self, k: int = 4, distance: str = "euclidean", rng=Random(123456), tries: int = 10):
        self.k = k
        self.distance = distance
        self.rng = rng
        # New attribute that specifies the number of times we will execute the algorithm, to avoid a possible bad rng initialization
        self.tries = tries

    def fit(self, observations):
        
        # Varibles to control the best try (minimizing total distance)
        best_total_distance = float('inf') # Worst result possible
        best_centroids = []
        best_assignments = []
        best_distances = []
        
        # We repeat the algorithm as much as it is specified to search the best result
        for n_try in range(self.tries):

            #Random initialization
            # We select K random points from the dataset as centroids
            index_inicials = self.rng.sample(range(len(observations)), self.k)

            current_centroids = []

            for index in index_inicials:
                # Copy the list so as not to modify the original
                current_centroids.append(observations[index]) 

            # Set a safety limit for the loop in case the algorithm doesn't converge (centroids keep moving)
            max_iterations = 100

            for iteration in range(max_iterations):
                # Assign each point to the closest centroid
                # Variables for this specific iteration
                new_assignments = []
                current_distances = []
                total_distance_sum = 0

                # For each point in the data, we calculate the distance to every centroid
                for point in observations:
                    min_distance = float('inf')
                    index_closest_centroid = -1

                    for i in range(self.k):
                        # Calculate distance depending on the method defined
                        sum_squares = 0
                        for d in range(len(point)):
                            sum_squares += (point[d] - current_centroids[i][d]) ** 2

                        current_distance = 0

                        if self.distance == "euclidean":
                            current_distance = math.sqrt(sum_squares)
                        elif self.distance == "squared-euclidean":
                            current_distance = sum_squares

                        # We identify the closest centroid and assign the point to that cluster
                        if current_distance < min_distance:
                            min_distance = current_distance
                            index_closest_centroid = i

                    new_assignments.append(index_closest_centroid)
                    total_distance_sum += min_distance
                    current_distances.append(min_distance)

                # Recalculate the centroids
                new_centroids = []
                for i in range(self.k):
                    cluster_points = [observations[idx] for idx, val in enumerate(new_assignments) if val == i]

                    if not cluster_points:
                        # If cluster is empty, we keep the old one to not crash
                        new_centroids.append(current_centroids[i])
                        continue

                    # We calculate the mean of all points in the cluster to find new center
                    dimension = len(cluster_points[0])
                    new_center = [sum(p[d] for p in cluster_points) / len(cluster_points) for d in range(dimension)]
                    new_centroids.append(new_center)

                # Check convergence (if they don't move, we exit)
                if new_centroids == current_centroids:
                    break

                current_centroids = new_centroids

            # If this execution resulted in a lower total distance than the previous best, we update it
            if total_distance_sum < best_total_distance:
                best_total_distance = total_distance_sum
                best_centroids = current_centroids
                best_assignments = new_assignments
                best_distances = current_distances

        # Finally, we store the best results in the attributes
        self.centroids_ = best_centroids
        self.X_assignments_ = best_assignments
        self.distances_ = best_distances
        return self


###############################################
#                 CLI Code                    #
###############################################


def main(args):
    # Set the random generator
    rng = Random(args.seed)

    # Load the dataset (With the ignore feature for the reading)
    dataset = read_csv(args.dataset, ignore_first = True, ignore_col = True)

    # Instantiate KMeans
    kmeans = KMeans(k=args.k, distance=args.distance, rng=rng, tries = args.tries)

    # Train the clustering model
    kmeans.fit(dataset)

    # Print some metrics
    print("Distances:", kmeans.distances_)
    print("Sum of distances:", sum(kmeans.distances_) if kmeans.distances_ else 0)
    print("Centroid positions:", kmeans.centroids_)
    print("Centroids assignments:", kmeans.X_assignments_)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset", type=str, help="Path to the CSV file containing the dataset."
    )
    parser.add_argument(
        "--k", type=int, default=4, help="Value for the 'k' parameter of KMeans."
    )
    parser.add_argument(
        "--distance",
        type=str,
        choices=["euclidean", "squared-euclidean"],
        default="euclidean",
        help="Distance metric used by KMeans.",
    )
    parser.add_argument("--seed", type=int, default=123456, help="RNG Seed.")

    # Inclusion of the new att. to the parse func.
    parser.add_argument(
        "--tries", type=int, default=10, help="Number of times to run the algorithm."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
