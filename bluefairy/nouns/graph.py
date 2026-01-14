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


def extract_similarity_clusters(g: nx.Graph) -> list[set[PRED_KEY]]:
    """
    Extracts similarity clusters from a similarity graph.
    :param g: the similarity graph
    :return: a list of sets, each set representing a cluster of similar predicates
    """
    return [set(c) for c in nx.connected_components(g)]


def select_representative_by_frequency(
    cluster: set[PRED_KEY],
    occurrences: dict[PRED_KEY, int],
    sim_matrix: Optional[pd.DataFrame] = None
) -> PRED_KEY:
    """
    Selects a representative predicate from a cluster based on frequency and similarity.
    :param cluster: the cluster of predicates
    :param occurrences: a dictionary of predicate occurrences
    :param sim_matrix: the similarity matrix as a DataFrame
    :return: the selected representative predicate
    """
    max_freq = max(occurrences.get(p, 0) for p in cluster)

    candidates = [
        p for p in cluster
        if occurrences.get(p, 0) == max_freq
    ]

    if len(candidates) == 1 or sim_matrix is None:
        return candidates[0]

    scores: dict[PRED_KEY, float] = {}

    for p in candidates:
        others = set(candidates) - {p}
        scores[p] = (
            sim_matrix[p][list(others)].mean()
            if others else 0.0
        )

    return max(scores, key=scores.get)


def build_unification_map(
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

    clusters = extract_similarity_clusters(g)

    unification_map: dict[PRED_KEY, PRED_KEY] = {}

    for cluster in clusters:
        representative = select_representative_by_frequency(
            cluster=cluster,
            occurrences=occurrences,
            sim_matrix=sim_matrix
        )
        for member in cluster:
            unification_map[member] = representative

    return unification_map
