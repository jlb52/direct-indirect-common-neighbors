"""
This module provides functions for evaluating direct and indirect 
common neighbor-based link prediction methods in network graphs using NetworkX. 

It includes functionality for converting graphs into a format suitable for 
computation and applying the link prediction algorithms. The main method, 
`dicn`, calculates scores for pairs of nodes, 
combining both first and second order neighbor information.

Functions in this module:
- dicn: Main function to compute link prediction scores.
- _map_nodes_to_integers: Utility to convert node labels to integers.
- _remap_integers_to_nodes: Revert integers back to original node labels.
- _generate_neighborhood_vectors: Generate neighborhood vectors for the graph.
- _generate_union_neighborhood_set: Create a union set of neighborhoods for two nodes.
- _compute_correlation_coefficient: Compute the correlation coefficient between two nodes.
- _trim_vectors_to_union_set: Trim neighborhood vectors based on a union set.
- _calculate_numerator and _calculate_denominator: Helper functions for correlation calculation.

The module is designed for use with undirected and non-multigraph NetworkX graphs.
"""


from typing import Iterable, Tuple, Any, Iterator, Dict

import networkx as nx
from networkx.utils import not_implemented_for
from networkx.algorithms.link_prediction import _apply_prediction
import numpy as np


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def dicn(
    G: nx.Graph, ebunch: Iterable[Tuple[Any, Any]] = None
) -> Iterator[Tuple[Any, Any, float]]:
    """
    Calculate direct indirect common neighbors scores for each pair of nodes in G.

    Args:
        G (nx.Graph): A NetworkX graph.
        ebunch (Iterable[Tuple[Any, Any]], optional): An iterable of node pairs.

    Returns:
        Iterator[Tuple[Any, Any, float]]: An iterator of tuples representing the node pairs and their direct and indirect common neighbors score.
    """
    G, mapping = _map_nodes_to_integers(G)

    neighbor_vectors = _generate_neighborhood_vectors(G)
    correlations_coefficients = {}

    def predict(u, v, mapping, G, neighbor_vectors, correlations_coefficients):
        u, v = _remap_integers_to_nodes(mapping, u, v)
        if u == v:
            raise nx.NetworkXAlgorithmError("Self links are not supported")
        else:
            correlation_coefficient = _compute_correlation_coefficient(
                u, v, neighbor_vectors, correlations_coefficients
            )
            pred = (1 + sum(1 for _ in nx.common_neighbors(G, u, v))) * (
                1 + correlation_coefficient
            )
            return pred

    return _apply_prediction(
        G,
        lambda u, v: predict(
            u, v, mapping, G, neighbor_vectors, correlations_coefficients
        ),
        ebunch,
    )


def _map_nodes_to_integers(G: nx.Graph) -> Tuple[nx.Graph, Dict[Any, int]]:
    """
    Map the nodes of G to integers.

    Args:
        G (nx.Graph): A NetworkX graph.

    Returns:
        Tuple[nx.Graph, Dict[Any, int]]: The graph with integer-labeled nodes and the mapping from original nodes to integers.
    """
    mapping = dict(zip(G, range(0, sum(1 for _ in G.nodes()))))
    return nx.convert_node_labels_to_integers(G), mapping


def _remap_integers_to_nodes(
    mapping: Dict[Any, int], u: int, v: int
) -> Tuple[Any, Any]:
    """
    Remap integers back to their original nodes.

    Args:
        mapping (Dict[Any, int]): A dictionary mapping original nodes to integers.
        u (int): An integer representing a node.
        v (int): An integer representing a node.

    Returns:
        Tuple[Any, Any]: The original nodes corresponding to the integers u and v.
    """
    return mapping[u], mapping[v]


def _generate_neighborhood_vectors(G: nx.Graph) -> np.ndarray:
    """
    Generate neighborhood vectors for a graph.

    Args:
        G (nx.Graph): A NetworkX graph.

    Returns:
        np.ndarray: An array representing the neighborhood vectors of the graph.
    """
    adjacency_matrix = nx.to_numpy_array(G)
    second_order_paths = adjacency_matrix @ adjacency_matrix
    np.fill_diagonal(second_order_paths, 0)
    first_order_mask = adjacency_matrix > 0
    second_order_mask = (second_order_paths > 0) & ~first_order_mask
    common_neighbors_count = adjacency_matrix @ adjacency_matrix
    np.fill_diagonal(common_neighbors_count, 0)
    common_neighbors_count[first_order_mask] += 1
    neighborhood_vectors = np.where(
        first_order_mask, common_neighbors_count, 0
    ) + np.where(second_order_mask, common_neighbors_count, 0)
    np.fill_diagonal(neighborhood_vectors, adjacency_matrix.sum(axis=1))
    return neighborhood_vectors


def _generate_union_neighborhood_set(
    u_neighborhood_vector: np.ndarray, v_neighborhood_vector: np.ndarray
) -> np.ndarray:
    """
    Generate a set of union neighborhoods for two neighborhood vectors.

    Args:
        u_neighborhood_vector (np.ndarray): A neighborhood vector for node u.
        v_neighborhood_vector (np.ndarray): A neighborhood vector for node v.

    Returns:
        np.ndarray: An array representing the union set of the two neighborhood vectors.
    """
    return np.where((u_neighborhood_vector + v_neighborhood_vector) > 0)[0]


def _compute_correlation_coefficient(
    u: int,
    v: int,
    neighbor_vectors: np.ndarray,
    correlations_coefficients: Dict[Tuple[int, int], float],
) -> float:
    """
    Compute the correlation coefficient between two nodes.

    Args:
        G (nx.Graph): A NetworkX graph.
        u (int): An integer representing node u.
        v (int): An integer representing node v.
        neighbor_vectors (np.ndarray): An array of neighborhood vectors.
        correlations_coefficients (Dict[Tuple[int, int], float]): A dictionary to store correlation coefficients.

    Returns:
        float: The correlation coefficient between nodes u and v.
    """
    if (v, u) in correlations_coefficients:
        return correlations_coefficients[(v, u)]
    u_neighborhood_vector, v_neighborhood_vector = (
        neighbor_vectors[u],
        neighbor_vectors[v],
    )
    union_neighborhood_set = _generate_union_neighborhood_set(
        u_neighborhood_vector, v_neighborhood_vector
    )

    u_neighborhood_vector, v_neighborhood_vector = _trim_vectors_to_union_set(
        u_neighborhood_vector, v_neighborhood_vector, union_neighborhood_set
    )

    if len(union_neighborhood_set) == 0:
        return 0

    u_vector_average, v_vector_average = np.mean(u_neighborhood_vector), np.mean(
        v_neighborhood_vector
    )
    u_diff, v_diff = (
        u_neighborhood_vector - u_vector_average,
        v_neighborhood_vector - v_vector_average,
    )

    numerator = _calculate_numerator(u_diff, v_diff)
    denominator = _calculate_denominator(u_diff, v_diff)

    correlation_coefficient = 0 if denominator == 0 else numerator / denominator
    correlations_coefficients[(u, v)] = correlation_coefficient

    return correlation_coefficient


def _trim_vectors_to_union_set(
    u_vector: np.ndarray, v_vector: np.ndarray, union_set: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trim vectors to the union set of neighborhoods.

    Args:
        u_vector (np.ndarray): A neighborhood vector for node u.
        v_vector (np.ndarray): A neighborhood vector for node v.
        union_set (np.ndarray): An array representing the union set of neighborhoods.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The trimmed neighborhood vectors for nodes u and v.
    """
    return u_vector[union_set], v_vector[union_set]


def _calculate_numerator(u_diff: np.ndarray, v_diff: np.ndarray) -> float:
    """
    Calculate the numerator for the correlation coefficient calculation.

    Args:
        u_diff (np.ndarray): Difference vector for node u.
        v_diff (np.ndarray): Difference vector for node v.

    Returns:
        float: The numerator of the correlation coefficient calculation.
    """
    return np.sum(u_diff * v_diff)


def _calculate_denominator(u_diff: np.ndarray, v_diff: np.ndarray) -> float:
    """
    Calculate the denominator for the correlation coefficient calculation.

    Args:
        u_diff (np.ndarray): Difference vector for node u.
        v_diff (np.ndarray): Difference vector for node v.

    Returns:
        float: The denominator of the correlation coefficient calculation.
    """
    u_denominator_sq = np.sum(u_diff**2)
    v_denominator_sq = np.sum(v_diff**2)
    if u_denominator_sq == 0 or v_denominator_sq == 0:
        return 0
    u_denominator = np.sqrt(u_denominator_sq)
    v_denominator = np.sqrt(v_denominator_sq)
    return u_denominator * v_denominator
