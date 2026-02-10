from collections import defaultdict
from pathlib import Path

import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from bluefairy.grammar.utils import PRED_KEY
from bluefairy.nouns.embedding import generate_predicate_embedding_sentences, build_embedding_space, \
    build_predicate_similarity_matrix
from bluefairy.nouns.graph import build_predicate_unification_map
from bluefairy.nouns.lexical_metrics import build_predicate_lexical_similarity_matrix
from bluefairy.nouns.unification import collect_predicates, create_predicate_terms_matrices, create_predicate_arity_matrix, compute_similarity_scores
from evaluation.data import load_test_set
from evaluation.unification.figures.generate_pca_scatter_plots import plot_embedding_pca_2d, plot_embedding_pca_3d
from evaluation.unification.utils import initialize_components, create_predicate_merge_mappings, TRANSFORMER
from evaluation.unification.tables import PATH as TABLES_PATH

PATH = Path(__file__).parent.resolve()


def truncate_colormap(cmap, minval=0.3, maxval=1.0, n=256):
    return LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n))
    )


def generate_predicate_alpha_merging_plot(
        list_of_fol_formulae: list[str],
        alpha: float,
        thresholds: list[float],
        output_file: Path
) -> None:
    _predicates, _matrix = initialize_components(list_of_fol_formulae, alpha)

    number_of_merges = []
    number_of_clusters = []

    for threshold in thresholds:
        _map = build_predicate_unification_map(
            sim_matrix=_matrix,
            occurrences=_predicates,
            threshold=threshold
        )
        number_of_merges.append(sum(1 for k, v in _map.items() if k != v))
        # count a cluster only if it has more than one member
        number_of_clusters.append(len(set(x for x in _map.values() if list(_map.values()).count(x) > 1)))

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, number_of_merges, label='Number of Merges', marker='o')
    plt.plot(thresholds, number_of_clusters, label='Number of Clusters\nwith more than one element', marker='o')
    plt.title(f'Predicate Merging and Clustering (Alpha={alpha})')
    plt.xlabel('Similarity Threshold')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)


def generate_predicate_merging_plot(
        fol_formulae: list[str],
        alphas: list[float],
        thresholds: list[float],
        output_file: Path
) -> None:
    """
    Generate a plot showing the number of merges and number of clusters for
    multiple alpha values across a range of similarity thresholds.

    Alpha values are encoded via colour gradients and shown using two colour bars.
    """
    fig, ax = plt.subplots(figsize=(13, 7))

    _predicates = collect_predicates(fol_formulae)
    matrices = create_predicate_terms_matrices(fol_formulae)
    predicate_sentences = generate_predicate_embedding_sentences(matrices)
    arity_predicate_matrix = create_predicate_arity_matrix(list(_predicates.keys()))

    embedding_predicate_space = build_embedding_space(
        list(predicate_sentences.values()),
        lambda x: TRANSFORMER.encode(x, normalize_embeddings=False)
    )

    semantic_predicate_matrix_score = build_predicate_similarity_matrix(
        embedding_predicate_space,
        predicate_sentences
    )
    lexical_predicate_matrix_score = build_predicate_lexical_similarity_matrix(
        list(_predicates.keys())
    )

    # --- Normalisation for alpha → colour
    # (low alpha = darker colour → invert AFTER normalisation)
    alpha_norm = Normalize(
        vmin=min(alphas),
        vmax=max(alphas)
    )

    merge_cmap = truncate_colormap(plt.cm.Blues, 0.3, 1.0)
    cluster_cmap = truncate_colormap(plt.cm.Oranges, 0.3, 1.0)

    data = []

    for alpha in alphas:
        print(f"Processing alpha={alpha:.1f}...")

        _matrix = arity_predicate_matrix * compute_similarity_scores(
            semantic_predicate_matrix_score,
            lexical_predicate_matrix_score,
            alpha=alpha
        )

        number_of_merges = []
        number_of_clusters = []

        for threshold in thresholds:
            _map = build_predicate_unification_map(
                sim_matrix=_matrix,
                occurrences=_predicates,
                threshold=threshold
            )

            number_of_merges.append(
                sum(1 for k, v in _map.items() if k != v)
            )

            number_of_clusters.append(
                len(set(
                    x for x in _map.values()
                    if list(_map.values()).count(x) > 1
                ))
            )
            data.append([alpha, threshold, number_of_merges[-1], number_of_clusters[-1]])

        df_data = pd.DataFrame(data, columns=["Alpha", "Threshold", "NumMerges", "NumNonTrivialClusters"])
        df_data.to_csv(TABLES_PATH / 'predicate_merging_data.csv', index=False)

        colour_value = 1.0 - alpha_norm(alpha)

        ax.plot(
            thresholds,
            number_of_merges,
            color=merge_cmap(colour_value),
            linewidth=2
        )

        ax.plot(
            thresholds,
            number_of_clusters,
            linestyle="--",
            color=cluster_cmap(colour_value),
            linewidth=2
        )

    ax.set_title("Predicate Merging and Clustering Across Alpha Values")
    ax.set_xlabel("Similarity Threshold")
    ax.set_ylabel("Count")
    ax.grid(True)

    # --- Colour bars ---
    sm_merges = ScalarMappable(norm=alpha_norm, cmap=merge_cmap.reversed())
    sm_merges.set_array([])

    sm_clusters = ScalarMappable(norm=alpha_norm, cmap=cluster_cmap.reversed())
    sm_clusters.set_array([])

    fig.colorbar(sm_merges, ax=ax, pad=-0.05)
    fig.colorbar(sm_clusters, ax=ax, pad=0.05)


    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)


def plot_unification_bubble_cloud(
    unification_map: dict[PRED_KEY, PRED_KEY],
    output_file: Path,
    title: str = "Predicate Unification Clusters",
    size_scale: int = 600,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)

    rep_to_cluster: dict[PRED_KEY, set[PRED_KEY]] = defaultdict(set)
    for pred, rep in unification_map.items():
        rep_to_cluster[rep].add(pred)

    clusters = [
        (rep, members)
        for rep, members in rep_to_cluster.items()
        if len(members) > 1
    ]

    if not clusters:
        return

    reps = [rep for rep, _ in clusters]
    sizes = [len(members) for _, members in clusters]

    coords = rng.normal(0, 1, size=(len(reps), 2))
    node_sizes = [s * size_scale for s in sizes]

    plt.figure(figsize=(10, 8))

    plt.scatter(
        coords[:, 0],
        coords[:, 1],
        s=node_sizes,
        alpha=0.85,
        edgecolors="black"
    )

    for (x, y), rep, size in zip(coords, reps, sizes):
        plt.text(
            x,
            y,
            f"{rep[0]}/{rep[1]}\n|C|={size}",
            ha="center",
            va="center",
            fontsize=8
        )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


if __name__ == "__main__":
    test_set = load_test_set()
    fol_formulae = test_set['FOL'].tolist()

    alphas = [0.1 * i for i in range(0, 11)]
    alpha = 1
    thresholds = [0.5 + i * 0.025 for i in range(1, 21)]

    generate_predicate_merging_plot(
        fol_formulae=fol_formulae,
        alphas=alphas,
        thresholds=thresholds,
        output_file=PATH / 'predicate_merging_plot.pdf'
    )

    # Example of plotting a similarity graph
    predicates, matrix = initialize_components(fol_formulae, alpha)
    maps = create_predicate_merge_mappings(thresholds[:-1], predicates, matrix)

    for i, t in enumerate(thresholds[:-1]):
        plot_output_file = PATH / f'predicate_similarity_graph_a{int(alpha * 10)}_t{int(t * 1000)}.pdf'
        plot_unification_bubble_cloud(
            unification_map=maps[i],
            output_file=plot_output_file,
            title=f"Predicate Similarity Graph (Threshold={t:.3f})"
        )