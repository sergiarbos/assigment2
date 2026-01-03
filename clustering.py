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

            #Iincialitzacio aleatoria
            # We select K random points from the dataset as centroids
            index_inicials = self.rng.sample(range(len(observations)), self.k)

            current_centroids = []

            for index in index_inicials:
                #Copiem la llista per no modificar la original
                current_centroids.append(observations[index]) 

            #fiquem limit de seguretat per el bucle per si l'algorisme no convergeix (centroides es mouen tota l'estona)
            iteracions_maximes = 100

            for iteracio in range(iteracions_maximes):
                #Assignem cada punt al centreide mes proper
                # Variables for this specific iteration
                nous_assignaments = []
                current_distances = []
                suma_distancies_total = 0

                # For each point in the data, we calculate the distance to every centroid
                for punt in observations:
                    distancia_minima = float('inf')
                    index_centreide_mes_proper = -1

                    for i in range(self.k):
                        #Calculem distancia depenent del mètode que s'ha definit
                        suma_quadrats = 0
                        for d in range(len(punt)):
                            suma_quadrats += (punt[d] - current_centroids[i][d]) ** 2

                        distancia_actual = 0

                        if self.distance == "euclidean":
                            distancia_actual = math.sqrt(suma_quadrats)
                        elif self.distance == "squared-euclidean":
                            distancia_actual = suma_quadrats

                        # We identify the closest centroid and assign the point to that cluster
                        if distancia_actual < distancia_minima:
                            distancia_minima = distancia_actual
                            index_centreide_mes_proper = i

                    nous_assignaments.append(index_centreide_mes_proper)
                    suma_distancies_total += distancia_minima
                    current_distances.append(distancia_minima)
                
                # Recalculem els centroides
                nous_centroides = []
                for i in range(self.k):
                    punts_cluster = [observations[idx] for idx, val in enumerate(nous_assignaments) if val == i]

                    if not punts_cluster:
                        # If cluster is empty, we keep the old one to not crash
                        nous_centroides.append(current_centroids[i])
                        continue

                    # We calculate the mean of all points in the cluster to find new center
                    dimensio = len(punts_cluster[0])
                    nou_centre = [sum(p[d] for p in punts_cluster) / len(punts_cluster) for d in range(dimensio)]
                    nous_centroides.append(nou_centre)

                # Comprovem convergència (si no es mouen, sortim)
                if nous_centroides == current_centroids:
                    break

                current_centroids = nous_centroides
            
            # If this execution resulted in a lower total distance than the previous best, we update it
            if suma_distancies_total < best_total_distance:
                best_total_distance = suma_distancies_total
                best_centroids = current_centroids
                best_assignments = nous_assignaments
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
