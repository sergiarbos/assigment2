from __future__ import annotations
import argparse
from utils import read_sms, split_observations_and_labels
from random import Random
import math


def tokenize_sms(message):
    puntuacio = ".,!?:;()\"'" #COM SEPAREM LES PARAULES
    tokens = [] 
    for word in message.split():
        clean_word = word.strip(puntuacio).lower()
        if len(clean_word) > 0:
            tokens.append(clean_word)
    return tokens


class MultinomialNaiveBayesClassifier:
    def __init__(self, assumed_probability=1):
        self.assumed_probability = assumed_probability
        self.word_count_class = {}
        self.total_word_count = {}
        self.docs_per_class = {}
        self.total_docs = 0
        self.vocabulary = set()
        

    def fit(self, observations, labels):
        self.total_docs = len(labels)
        for i in range(len(observations)):
            message = observations[i]
            category = labels[i]

            self.docs_per_class[category] = self.docs_per_class.get(category, 0) + 1
            if category not in self.word_count_class:
                self.word_count_class[category] = {}

            for word in message:
                self.vocabulary.add(word)
                self.word_count_class[category][word] = self.word_count_class[category].get(word, 0) + 1
                self.total_word_count[category] = self.total_word_count.get(category, 0) + 1

        return self
            

    def predict(self, observations):
        predictions = []

        for message in observations:
            best_category = None
            max_log_prob = float('-inf') #valor mes baix posible

            for category in self.docs_per_class:
                prior_prob = self.docs_per_class[category] / self.total_docs
                log_prob = math.log(prior_prob)

                for word in message:
                    if word in self.vocabulary:
                        # vegades que la paraula A apareix a la categoria B / total de missatges a la categoria B
                        time_in_class = self.word_count_class[category].get(word, 0)
                        basic_prob = time_in_class / self.docs_per_class[category]
                        count_a = sum(self.word_count_class[c].get(word, 0) for c in self.docs_per_class)

                        # Fórmula: (pes * probabilitat_assumida + recompte(A) * p(A|B)) / (recompte(A) + pes)
                        weight = 1.0
                        num = (weight * self.assumed_probability) + (count_a * basic_prob)
                        den = count_a + weight
                        weighted_p = num / den

                        # Sumar al total (equivalent a multiplicar probabilitats) 
                        log_prob += math.log(weighted_p)

                if log_prob > max_log_prob:
                    max_log_prob = log_prob
                    best_category = category

            predictions.append(best_category)

        return predictions
    
    def score(self, data, labels) -> float:
        predicted = self.predict(data)
        correct = sum(
            1 if pred == expected else 0 for pred, expected in zip(predicted, labels)
        )
        return correct / len(data)


###############################################
#                 CLI Code                    #
###############################################


def main(args):
    # Set the random generator
    rng = Random(args.seed)

    # Load the dataset
    messages, labels = read_sms(args.dataset)

    # Tokenize the messages
    tokenized_messages = []
    for msg in messages:
        tokenized_messages.append(tokenize_sms(msg))

    # Split the dataset into training and test sets
    # NOTE: consider args.test_ratio and args.seed
    indices = list(range(len(tokenized_messages)))
    rng.shuffle(indices)

    #Calcular el punt de divisio basant-nos en el ratio de test
    split_idx = int(len(indices)*(1-args.test_ratio))

    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    # Creem les llistes d'entrenament per a observacions i etiquetes
    train_obs = []
    for i in train_indices:
        train_obs.append(tokenized_messages[i])

    train_labels = []
    for i in train_indices:
        train_labels.append(labels[i])

    # Creem les llistes de test per a observacions i etiquetes
    test_obs = []
    for i in test_indices:
        test_obs.append(tokenized_messages[i])

    test_labels = []
    for i in test_indices:
        test_labels.append(labels[i])

    # Instantiate the decision tree classifier
    mnb = MultinomialNaiveBayesClassifier(assumed_probability=args.assumed_probability)

    # Train the classifier using the training data
    print(f"Training model with {len(train_obs)} observations ...")
    mnb.fit(train_obs, train_labels)

    # Predict over the test set
    print("Evaluating test set...")
    

    # Evaluate these predictions using the accuracy score and print the information
    # Calculem la precisió final comparant prediccions amb etiquetes reals
    accuracy = mnb.score(test_obs, test_labels)
    print(f"\nResults for {args.dataset}:")
    print(f"Accuracy: {accuracy * 100:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset", type=str, help="Path to the CSV file containing the dataset."
    )
    parser.add_argument(
        "--assumed_probability",
        type=int,
        default=1,
        help="Value for the 'assumed_probability' parameter.",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.3, help="Ratio for the test set split."
    )
    parser.add_argument("--seed", type=int, default=123456, help="RNG Seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
