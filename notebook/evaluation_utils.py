"""
This module provides functionalities for evaluating various link prediction methods on network graphs.

It includes functions for loading graph datasets using PyTorch Geometric, converting them into NetworkX graphs, and evaluating different link prediction algorithms by splitting the edges into training and testing sets. The evaluation results are summarized using AUC scores and computation times. 

Key functionalities include:
- Loading and converting graph datasets
- Splitting graph edges for training and testing
- Evaluating link prediction methods such as Resource Allocation, Jaccard Coefficient, etc.
- Summarizing the evaluation results in a pandas DataFrame

Dependencies:
- NetworkX for graph manipulation and analysis
- PyTorch Geometric for loading graph datasets
- Pandas for data summarization
- Scikit-learn for AUC score calculation

Example usage:
    G = load_dataset_and_convert_to_networkx('Cora')
    results = evaluate_link_prediction_methods(G, methods_dict)
    summary_df = summarize_evaluation_results(results)
"""

import random
import time
import pandas as pd
import networkx as nx
import torch_geometric.datasets as pyg_datasets
from torch_geometric.utils import to_networkx
from sklearn.metrics import roc_auc_score


def load_dataset_and_convert_to_networkx(name):
    """
    Load a dataset from PyTorch Geometric and convert it to a NetworkX graph.

    Args:
        name (str): Name of the dataset.

    Returns:
        networkx.Graph: The loaded graph.
    """
    dataset = pyg_datasets.Planetoid(root="/tmp/" + name, name=name)
    data = dataset[0]
    G = to_networkx(data, to_undirected=True)
    return G


def _split_edges(G, train_percent):
    """
    Split the edges of the graph into training and testing sets.

    Args:
        G (networkx.Graph): The graph.
        train_percent (float): The proportion of edges to include in the training set.

    Returns:
        tuple: Training and testing edges.
    """
    edges = list(G.edges())
    random.shuffle(edges)
    num_train = int(train_percent * len(edges))
    train_edges = edges[:num_train]
    test_edges = edges[num_train:]
    return train_edges, test_edges


def _non_existing_edges(G, num_edges):
    """
    Generate a list of non-existing edges.

    Args:
        G (networkx.Graph): The graph.
        num_edges (int): Number of non-existing edges to generate.

    Returns:
        list: A list of non-existing edges.
    """
    non_edges = list(nx.non_edges(G))
    random.shuffle(non_edges)
    return non_edges[:num_edges]


def _calculate_auc(scores):
    """
    Calculate the AUC score from given scores.

    Args:
        scores (list): A list of tuples (true label, predicted score).

    Returns:
        float: The AUC score.
    """
    y_true, y_scores = zip(*scores)
    return roc_auc_score(y_true, y_scores)


def _evaluate_method(G, method_func, test_edges, test_labels):
    """
    Evaluate a link prediction method.

    Args:
        G (networkx.Graph): The graph.
        method_func (callable): The link prediction method.
        test_edges (list): List of edges to test.
        test_labels (list): List of labels corresponding to test edges.

    Returns:
        tuple: AUC score and execution time for the method.
    """
    G_train = G.copy()
    G_train.remove_edges_from(test_edges)

    test_pairs = test_edges + _non_existing_edges(G_train, len(test_edges))

    start_time = time.time()
    scores = [
        (label, score)
        for (u, v, score), label in zip(method_func(G_train, test_pairs), test_labels)
    ]
    end_time = time.time()

    auc_score = _calculate_auc(scores)
    execution_time = end_time - start_time

    return auc_score, execution_time


def evaluate_link_prediction_methods(G, methods, splits, repetitions=5):
    """
    Evaluate multiple link prediction methods on a graph.

    Args:
        G (networkx.Graph): The graph.
        methods (dict): A dictionary of link prediction methods.
        splits (list): A list of splits for training and testing.
        repetitions (int): Number of repetitions for evaluation.

    Returns:
        dict: A dictionary of evaluation results.
    """
    results = {
        method: {split: {"AUC": [], "Time": []} for split in splits}
        for method in methods.keys()
    }

    for method_name, method_func in methods.items():
        for split in splits:
            for _ in range(repetitions):
                _, test_edges = _split_edges(G, split)
                test_labels = [1] * len(test_edges) + [0] * len(
                    _non_existing_edges(G, len(test_edges))
                )

                auc_score, execution_time = _evaluate_method(
                    G, method_func, test_edges, test_labels
                )

                results[method_name][split]["AUC"].append(auc_score)
                results[method_name][split]["Time"].append(execution_time)

    return results


def summarize_and_format_results(results):
    """
    Summarize and reformat evaluation results into a pandas dataframe with multi-level columns.
    AUC scores are presented as percentages and Time is in seconds. The DataFrame is structured for easier comparison.

    Args:
        results (dict): The results from evaluate_link_prediction_methods function.

    Returns:
        pandas.DataFrame: A reformatted dataframe summarizing the results.
    """
    formatted_results = {("AUC Mean (%)", method): [] for method in results}
    formatted_results.update({("Time Mean (s)", method): [] for method in results})
    splits = []

    for split in next(iter(results.values())).keys():
        splits.append(split)
        for method in results:
            auc_mean = (
                pd.Series(results[method][split]["AUC"]).mean() * 100
            )  # Convert to percentage
            time_mean = round(
                pd.Series(results[method][split]["Time"]).mean(), 2
            )  # Truncate time

            formatted_results[("AUC Mean (%)", method)].append(f"{auc_mean:.1f}%")
            formatted_results[("Time Mean (s)", method)].append(time_mean)

    df = pd.DataFrame(formatted_results, index=splits)
    df.index.name = "Split"
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    return df
