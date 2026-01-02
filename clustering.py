import argparse
import math
from random import Random
from utils import read_csv


class KMeans:
    def __init__(self, k: int = 4, distance: str = "euclidean", rng=Random(123456)):
        self.k = k
        self.distance = distance
        self.rng = rng

    def fit(self, observations):
        #Iincialitzacio aleatoria
        index_inicials = self.rng.sample(range(len(observations)), self.k)
        self.centroids_ = []
        for index in index_inicials:
            #Copiem la llista per no modificar la original
            self.centroids_.append(observations[index])
        
        #fiquem limit de seguretat per el bucle
        iteracions_maximes = 100
        for iteracio in range(iteracions_maximes):
            #Assignem cada punt al centreide mes proper
            nous_assignaments = []
            suma_distancies_total = 0
            for punt in observations:
                distancia_minima = float('inf')
                index_centreide_mes_proper = -1
                for i in range(self.k):
                    #Calculem distancia
                    suma_quadrats = 0
                    for d in range(len(punt)):
                        suma_quadrats += (punt[d] - self.centroids_[i][d]) ** 2
                    distancia_actual = 0
                    if self.distance == "euclidean":
                        distancia_actual = math.sqrt(suma_quadrats)
                    elif self.distance == "squared-euclidean":
                        distancia_actual = suma_quadrats

                    if distancia_actual < distancia_minima:
                        distancia_minima = distancia_actual
                        index_centreide_mes_proper = i
                nous_assignaments.append(index_centreide_mes_proper)
                suma_distancies_total += distancia_minima
            
            #Recalculem els centroides
            nous_centroides = []


        self.distances_ = []
        self.X_assignments_ = []
        return self


###############################################
#                 CLI Code                    #
###############################################


def main(args):
    # Set the random generator
    rng = Random(args.seed)

    # Load the dataset
    dataset = read_csv(args.dataset)

    # Instantiate KMeans
    kmeans = KMeans(k=args.k, distance=args.distance, random_state=rng)

    # Train the clustering model
    """YOUR CODE HERE"""

    # Print some metrics
    print("Distances:", ...)
    print("Sum of distances:", ...)
    print("Centroid positions:", ...)
    print("Centroids assignments:", ...)


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
