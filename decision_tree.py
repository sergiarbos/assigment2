from __future__ import annotations
from math import log
from dataclasses import dataclass
from typing import Optional
import argparse
from utils import read_csv, split_observations_and_labels
from random import Random
from itertools import combinations


def gini(labels) -> float:
    total = len(labels)
    results = _unique_counts(labels)
    imp = 1

    for label, count in results.items():
        prob = count / total
        imp -= prob**2

    return imp


def entropy(labels):
    total = len(labels)
    results = _unique_counts(labels)
    imp = 0

    for label, count in results.items():
        prob = count / total
        imp -= prob * _log2(prob)

    # File error?
    return imp


class DecisionTreeClassifier:
    def __init__(self, scoref=gini, beta=0, prune_threshold=0):
        self.scoref = scoref
        self.beta = beta
        self.prune_threshold = prune_threshold

    def fit(self, observations, labels):
        self._iterative_build_tree(observations, labels)
        self._prune_tree()
        return self

    def predict(self, observations):
        labels = []
        for observation in observations:
            leaf = self.tree_.follow_tree(observation)
            label = max(leaf.results.items(), key=lambda x: x[1])[0]
            labels.append(label)
        return labels

    def score(self, data, labels) -> float:
        predicted = self.predict(data)
        correct = sum(
            1 if pred == expected else 0 for pred, expected in zip(predicted, labels)
        )
        return correct / len(data)

    def _iterative_build_tree(self, observations, labels):
        self.tree_ = None

        # Firstly, we initialize the stack with all the raw data to create the first distinction (node), so it has no parent and is the root.
        stack = [(observations, labels, None, "root")]

        while stack:
            current_obs, current_labels, parent, branch = stack.pop()
            
            # Purity of the current data
            current_score = self.scoref(current_labels)
            
            best_gain = 0.0
            best_criteria = None
            best_sets = None

            # If it is not pure, we seek for the best separation point
            if current_score > 0:
                num_columns = len(current_obs[0])
                
                # For each attribute
                for col in range(num_columns):
                    
                    # We list the values of the given set
                    unique_vals = list(_unique_values(current_obs, col))
                    candidates = []

                    # If the column first elem is numeric (we assume all the rest will be, it is the common thing), we just continue and the divideset will throw == or >< questions
                    if not unique_vals or _is_numeric(unique_vals[0]):
                        candidates = unique_vals
                    else:
                        """(i.e., consider queries of ∈ instead of =)"""
                        # For categoricals, we generate combinations of values to consider subsets (is red or green?)
                        # We stop at half the size because splitting a group from the rest is the same as splitting the rest from that group, we avoid calculating the same twice
                        for r in range(1, (len(unique_vals) // 2) + 1):
                            for subset in combinations(unique_vals, r):
                                candidates.append(list(subset))
                    
                    for value in candidates:

                        # Divideset asks a yes or no question/condition and divides the column in two groups (the ones who satisfies the cond. and the ones who don't)
                        set1, labels1, set2, labels2 = _divideset(current_obs, current_labels, col, value)

                        # If one of the groups is empty, this separation gives us no value
                        if not set1 or not set2:
                            continue

                        # Now we calculate how much do we really gain with this distinction from our initial state, with the information gain formula
                        # Variable to controle the "weight" of each group, because it could be dispair in number and with this we ensure a correct representation of the gains. (And it is how the formula works)
                        w = len(labels1) / len(current_labels)
                        gain = current_score - w * self.scoref(labels1) - (1 - w) * self.scoref(labels2)

                        # If the result is better than what we had, we shall store it, alongside witg the data (the condition (value), which column, the sets we got...)
                        if gain > best_gain:
                            best_gain = gain
                            best_criteria = (col, value)
                            best_sets = ((set1, labels1), (set2, labels2))

            # After scanning all the columns and obtaining the best separation, we must decide wether we set it as a leaf or just a node to continue the process
            # If the gain doesn't surpass Beta (criteria defined in the project) we will leave it a leaf bc it is not worth it dividing

            if best_gain > self.beta and best_sets is not None:
                # Dividing node case
                col, val = best_criteria
                node = Node.new_node(col, val, None, None)
                
                (true_obs, true_labels), (false_obs, false_labels) = best_sets
                
                # We add the new nodes to the stack for them to be processed later in the loop
                stack.append((false_obs, false_labels, node, "false"))
                stack.append((true_obs, true_labels, node, "true"))
            else:
                # Leaf case
                node = Node.new_leaf(current_labels)

            # We shall connect the node with it's parent
            if parent is None:
                # Was alr the root
                self.tree_ = node
            elif branch == "true":
                # If it came from true branch
                parent.true_branch = node
            elif branch == "false":
                # If it came from false branch
                parent.false_branch = node

    def _prune_tree(self):
        """YOUR CODE HERE"""
        raise NotImplementedError("TODO")


@dataclass
class Node:
    column: Optional[int]
    value: Optional[int | float | str]
    results: Optional[dict[int | float | str, int]]
    true_branch: Optional[Node]
    false_branch: Optional[Node]

    def is_leaf(self):
        return self.true_branch is None

    @classmethod
    def new_node(cls, column, value, true_branch, false_branch):
        """Create a new instance of this class representing a decision node."""
        return cls(column, value, None, true_branch, false_branch)

    @classmethod
    def new_leaf(cls, labels):
        """Create a new instance of this class representing a leaf."""
        return cls(None, None, _unique_counts(labels), None, None)

    def print_tree(self, indent=""):
        """Prints to stdout a representation of the tree."""
        if self.is_leaf():
            print(self.results)
        else:
            # Print the criteria
            if _is_numeric(self.value):
                print(f"{self.column}: <= {self.value}?")
            
            # To print the ∈ queries
            elif isinstance(self.value, (list, tuple)):
                print(f"{self.column} in {self.value}?")
            else:
                print(f"{self.column}: {self.value}?")
            # Print the branches
            print(f"{indent}T->", end="")
            self.true_branch.print_tree(indent + " ")
            print(f"{indent}F->", end="")
            self.false_branch.print_tree(indent + " ")

    def follow_tree(self, observation):
        """
        Traverse the (sub)tree by answering the queries, until a leaf is reached.

        This method returns the leaf that this observation reaches.
        """
        current = self
        while not current.is_leaf():
            query_fn = _get_query_fn(current.column, current.value)
            current = (
                current.true_branch if query_fn(observation) else current.false_branch
            )

        return current


###############################################
#             UTILITY FUNCTIONS               #
###############################################


def _unique_counts(values):
    """Count how many times each value appears in `values`"""
    results = {}
    for value in values:
        if value not in results:
            results[value] = 1
        else:
            results[value] += 1
    return results


def _is_numeric(value):
    """Checks if a value is numeric (i.e. a float or an int)"""
    return isinstance(value, int) or isinstance(value, float)


def _get_query_fn(column, value):
    """
    Create a function that separates observations based on a query.
    The query can be:

    a) categorical: the created function returns true
       iff. the observation has the exact value in the column specified.
    b) continuous: the created function returns true
       iff. the observation has a value smaller or equal than the
       reference one in the column specified.

    Note: consider any column with a numeric value as continuous.
    """
    if _is_numeric(value):
        return lambda prot: prot[column] <= value
    elif isinstance(value, (list, tuple, set)):
        # Modification to consider all the subsets of categorical values "(i.e., consider queries of ∈ instead of =)"
        return lambda prot: prot[column] in value
    else:
        return lambda prot: prot[column] == value


def _unique_values(table, column_idx):
    """Returns a set of the values in the columns of a table."""
    values = set()
    for row in table:
        values.add(row[column_idx])
    return values


def _log2(x):
    return log(x) / log(2)


def _divideset(observations, labels, column, value):
    """
    Divides a set on a specific column.
    Can handle numeric or categorical values
    """
    query_fn = _get_query_fn(column, value)

    observations1, labels1, observations2, labels2 = [], [], [], []

    for row, label in zip(observations, labels):
        if query_fn(row):
            observations1.append(row)
            labels1.append(label)
        else:
            observations2.append(row)
            labels2.append(label)

    return observations1, labels1, observations2, labels2


###############################################
#                 CLI Code                    #
###############################################


def main(args):
    # Set the random generator
    rng = Random(args.seed)

    # Load the dataset 
    # (We ignore the first row)
    dataset = read_csv(args.dataset, ignore_first = True)
    observations, labels = split_observations_and_labels(dataset)

    # Split the dataset into training and test sets
    # NOTE: consider args.test_ratio and args.seed
    
    # As many idxs as observations are needed
    indexs = list(range(len(observations)))
    rng.shuffle(indexs)

    # After the shuffle, we rearrange the obs and the labels in new lists
    shuffled_obs = [observations[i] for i in indexs]
    shuffled_labels = [labels[i] for i in indexs]

    # We get the "frontier" for the training obs and the test obs
    split_idx = int(len(observations) * (1 - args.test_ratio))

    # With the frontier, we classify the train and testing labels and obs
    train_observations = shuffled_obs[:split_idx]
    train_labels = shuffled_labels[:split_idx]
    
    test_observations = shuffled_obs[split_idx:]
    test_labels = shuffled_labels[split_idx:]

    # We get the function
    if args.scoref == "entropy":
        score_function = entropy 
    else:
        score_function = gini

    # Instantiate the decision tree classifier
    dec_tree = DecisionTreeClassifier(
        scoref=score_function, beta=args.beta, prune_threshold=args.prune_threshold
    )

    # Train the decision tree using the training data
    print(f"Training tree with {len(train_observations)} observations...")
    dec_tree.fit(train_observations, train_labels)

    # Print the tree structure
    print("Tree Structure:")
    dec_tree.tree_.print_tree()

    # Predict over the test set
    predictions = dec_tree.predict(test_observations)

    # Evaluate these predictions using the accuracy score and print the information
    accuracy = dec_tree.score(test_observations, test_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset", type=str, help="Path to the CSV file containing the dataset."
    )
    parser.add_argument(
        "--scoref",
        type=str,
        choices=["gini", "entropy"],
        default="gini",
        help="Impurity measure criterion for the decision tree.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="Value for the 'beta' parameter in the decision tree.",
    )
    parser.add_argument(
        "--prune-threshold", type=float, default=0.0, help="Prune threshold."
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.3, help="Ratio for the test set split."
    )
    parser.add_argument("--seed", type=int, default=123456, help="RNG Seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
