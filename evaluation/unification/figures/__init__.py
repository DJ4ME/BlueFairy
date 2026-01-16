from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

from bluefairy.grammar.utils import PRED_KEY
from bluefairy.nouns.graph import build_predicate_unification_map, build_similarity_graph, \
    select_representative_by_frequency
from bluefairy.nouns.unification import generate_predicate_similarity_matrix, collect_predicates, \
    create_predicate_terms_matrices
from evaluation.data import load_test_set

PATH = Path(__file__).parent.resolve()

def generate_predicate_alpha_merging_plot(
        list_of_fol_formulae: list[str],
        alpha: float,
        thresholds: list[float],
        output_file: Path
) -> None:
    predicates = collect_predicates(list_of_fol_formulae)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    matrices = create_predicate_terms_matrices(list_of_fol_formulae)
    matrix = generate_predicate_similarity_matrix(predicates, matrices, model, alpha)

    number_of_merges = []
    number_of_clusters = []

    for threshold in thresholds:
        map = build_predicate_unification_map(
            sim_matrix=matrix,
            occurrences=predicates,
            threshold=threshold
        )
        number_of_merges.append(sum(1 for k, v in map.items() if k != v))
        # count a cluster only if it has more than one member
        number_of_clusters.append(len(set(x for x in map.values() if list(map.values()).count(x) > 1)))

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, number_of_merges, label='Number of Merges', marker='o')
    plt.plot(thresholds, number_of_clusters, label='Number of Clusters\nwith more than one element', marker='o')
    plt.title(f'Predicate Merging and Clustering (Alpha={alpha})')
    plt.xlabel('Similarity Threshold')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)


def build_cluster_summary_graph(
    clusters: list[set[PRED_KEY]],
    occurrences: dict[PRED_KEY, int]
) -> nx.Graph:
    g = nx.Graph()

    for cluster in clusters:
        if len(cluster) <= 1:
            continue

        rep = select_representative_by_frequency(cluster, occurrences)
        g.add_node(
            rep,
            size=len(cluster),
            members=cluster
        )

    return g



def plot_similarity_graph(
    g: nx.Graph,
    output_file: Path,
    title: str = "Predicate Unification Clusters",
    size_scale: int = 400,
):
    plt.figure(figsize=(10, 8))

    pos_init = {
        n: (np.random.normal(), np.random.normal())
        for n in g.nodes
    }

    pos = nx.spring_layout(
        g,
        pos=pos_init,
        seed=42,
        k=1.5,
        iterations=100
    )

    node_sizes = [
        g.nodes[n].get("size", 1) * size_scale
        for n in g.nodes
    ]

    labels = {
        n: f"{n[0]}/{n[1]}\n|C|={g.nodes[n].get('size', 1)}"
        for n in g.nodes
    }

    nx.draw_networkx_nodes(
        g,
        pos,
        node_size=node_sizes,
        node_color="#6BAED6",
        edgecolors="black",
        alpha=0.85,
    )

    nx.draw_networkx_labels(
        g,
        pos,
        labels=labels,
        font_size=8,
    )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


if __name__ == "__main__":
    test_set = load_test_set()
    fol_formulae = test_set['FOL'].tolist()

    alpha = 0.1
    thresholds = [0.5 + i * 0.025 for i in range(1, 21)]

    #output_file = PATH / f'predicate_alpha_{int(alpha * 100)}_merging_plot.pdf'
    #generate_predicate_alpha_merging_plot(
    #    list_of_fol_formulae=fol_formulae,
    #    alpha=alpha,
    #    thresholds=thresholds,
    #    output_file=output_file
    #)

    # Example of plotting a similarity graph
    predicates = collect_predicates(fol_formulae)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    matrices = create_predicate_terms_matrices(fol_formulae)
    matrix = generate_predicate_similarity_matrix(predicates, matrices, model, alpha)


    for t in thresholds[:-1]:
        g = build_similarity_graph(
            sim_matrix=matrix,
            threshold=t,
            directed=False
        )
        g = build_cluster_summary_graph(
            clusters=[set(c) for c in nx.connected_components(g)],
            occurrences=predicates
        )

        plot_output_file = PATH / f'predicate_similarity_graph_{int(t*1000)}.pdf'
        plot_similarity_graph(
            g=g,
            output_file=plot_output_file,
            title=f"Predicate Similarity Graph (Threshold={t:.3f})"
        )


