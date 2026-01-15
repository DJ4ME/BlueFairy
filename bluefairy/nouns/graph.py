import pandas as pd
import networkx as nx
from typing import Optional
from bluefairy.grammar.utils import PRED_KEY


def build_similarity_graph(
    sim_matrix: pd.DataFrame,
    threshold: float,
    directed: bool = False
) -> nx.Graph:
    """
    Builds a similarity graph from a similarity matrix.
    :param sim_matrix: the similarity matrix as a DataFrame
    :param threshold: the similarity threshold for creating edges
    :param directed: whether the graph is directed or not
    """
    g = nx.DiGraph() if directed else nx.Graph()

    labels = sim_matrix.index.tolist()
    g.add_nodes_from(labels)

    for i, u in enumerate(labels):
        for j, v in enumerate(labels):
            if i >= j:
                continue
            score = sim_matrix.iat[i, j]
            if score >= threshold:
                g.add_edge(u, v, weight=float(score))

    return g


def extract_predicate_similarity_clusters(g: nx.Graph) -> list[set[PRED_KEY]]:
    """
    Extracts similarity clusters from a similarity graph.
    :param g: the similarity graph
    :return: a list of sets, each set representing a cluster of similar predicates
    """
    return [set(c) for c in nx.connected_components(g)]


def extract_constant_similarity_clusters(g: nx.Graph) -> list[set[str]]:
    """
    Extracts similarity clusters from a similarity graph.
    :param g: the similarity graph
    :return: a list of sets, each set representing a cluster of similar constants
    """
    return [set(c) for c in nx.connected_components(g)]


def select_representative_by_frequency(
    cluster: set[str] or set[PRED_KEY],
    occurrences: dict[str] or dict[PRED_KEY, int],
    sim_matrix: Optional[pd.DataFrame] = None
) -> str or PRED_KEY:
    """
    Selects a representative from a cluster based on frequency and similarity.
    :param cluster: the cluster of items (strings or predicates)
    :param occurrences: a dictionary of item occurrences
    :param sim_matrix: the similarity matrix as a DataFrame
    :return: the selected representative item
    """
    max_freq = max(occurrences.get(c, 0) for c in cluster)

    candidates = [
        c for c in cluster
        if occurrences.get(c, 0) == max_freq
    ]

    if len(candidates) == 1 or sim_matrix is None:
        return candidates[0]

    scores: dict[str, float] = {}

    for c in candidates:
        others = set(candidates) - {c}
        scores[c] = (
            sim_matrix[c][list(others)].mean()
            if others else 0.0
        )

    return max(scores, key=scores.get)


def select_predicate_representative_by_frequency(
    cluster: set[PRED_KEY],
    occurrences: dict[PRED_KEY, int],
    sim_matrix: Optional[pd.DataFrame] = None
) -> PRED_KEY:
    """
    Selects a predicate representative predicate from a cluster based on frequency and similarity.
    :param cluster: the cluster of predicates
    :param occurrences: a dictionary of predicate occurrences
    :param sim_matrix: the similarity matrix as a DataFrame
    :return: the selected representative predicate
    """
    return select_representative_by_frequency(
        cluster=cluster,
        occurrences=occurrences,
        sim_matrix=sim_matrix
    )


def select_constant_representative_by_frequency(
    cluster: set[str],
    occurrences: dict[str, int],
    sim_matrix: Optional[pd.DataFrame] = None
) -> str:
    """
    Selects a constant representative constant from a cluster based on frequency and similarity.
    :param cluster: the cluster of constants
    :param occurrences: a dictionary of constant occurrences
    :param sim_matrix: the similarity matrix as a DataFrame
    :return: the selected representative constant
    """
    return select_representative_by_frequency(
        cluster=cluster,
        occurrences=occurrences,
        sim_matrix=sim_matrix
    )


def build_predicate_unification_map(
    sim_matrix: pd.DataFrame,
    occurrences: dict[PRED_KEY, int],
    threshold: float = 0.8
) -> dict[PRED_KEY, PRED_KEY]:
    """
    Builds a unification map from a similarity matrix and occurrences.
    :param sim_matrix: the similarity matrix as a DataFrame
    :param occurrences: a dictionary of predicate occurrences
    :param threshold: the similarity threshold for clustering
    :return: a dictionary mapping each predicate to its representative
    """
    g = build_similarity_graph(
        sim_matrix=sim_matrix,
        threshold=threshold,
        directed=False
    )

    clusters = extract_predicate_similarity_clusters(g)

    unification_map: dict[PRED_KEY, PRED_KEY] = {}

    for cluster in clusters:
        representative = select_predicate_representative_by_frequency(
            cluster=cluster,
            occurrences=occurrences,
            sim_matrix=sim_matrix
        )
        for member in cluster:
            unification_map[member] = representative

    return unification_map


def build_constant_unification_map(
    sim_matrix: pd.DataFrame,
    occurrences: dict[str, int],
    threshold: float = 0.8
) -> dict[str, str]:
    """
    Builds a unification map from a similarity matrix and occurrences.
    :param sim_matrix: the similarity matrix as a DataFrame
    :param occurrences: a dictionary of constant occurrences
    :param threshold: the similarity threshold for clustering
    :return: a dictionary mapping each constant to its representative
    """
    g = build_similarity_graph(
        sim_matrix=sim_matrix,
        threshold=threshold,
        directed=False
    )

    clusters = extract_constant_similarity_clusters(g)

    unification_map: dict[str, str] = {}

    for cluster in clusters:
        representative = select_constant_representative_by_frequency(
            cluster=cluster,
            occurrences=occurrences,
            sim_matrix=sim_matrix
        )
        for member in cluster:
            unification_map[member] = representative

    return unification_map
