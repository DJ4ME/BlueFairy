from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from bluefairy.grammar.utils import PRED_KEY
from bluefairy.nouns.embedding import generate_predicate_embedding_sentences, build_embedding_space, \
    build_predicate_similarity_matrix
from bluefairy.nouns.graph import build_predicate_unification_map
from bluefairy.nouns.unification import collect_predicates, create_predicate_terms_matrices, \
    create_predicate_arity_matrix, compute_similarity_scores
from evaluation.data import load_test_set
from evaluation.unification.utils import TRANSFORMER
import umap


PATH = Path(__file__).parent.resolve()


def plot_embedding_pca_2d(
    embedding_space: dict[str, object],
    mapping: dict[PRED_KEY, PRED_KEY],
    output_file: Path,
    cluster_size_threshold: int = 9
) -> None:
    x = embedding_space["embeddings"]
    labels = list(mapping.keys())

    # --- build clusters ---
    clusters: dict[PRED_KEY, list[PRED_KEY]] = {}
    for k, v in mapping.items():
        clusters.setdefault(v, []).append(k)

    # assign cluster ids only to clusters with size > cluster_size_threshold
    cluster_id = {}
    cid = 0
    for rep, members in clusters.items():
        if len(members) > cluster_size_threshold:
            for m in members:
                cluster_id[m] = cid
            cid += 1

    point_cluster_ids = [
        cluster_id.get(lbl, -1)
        for lbl in labels
    ]

    # --- PCA ---
    pca = PCA(n_components=2, random_state=0)
    x_pca = pca.fit_transform(x)

    plt.figure(figsize=(9, 7))

    # plot singletons first (grey)
    singleton_idxs = [i for i, c in enumerate(point_cluster_ids) if c == -1]
    plt.scatter(
        x_pca[singleton_idxs, 0],
        x_pca[singleton_idxs, 1],
        c="lightgrey",
        s=30,
        alpha=0.5,
    )

    # plot real clusters
    cmap = plt.cm.tab20
    for c in sorted(set(point_cluster_ids)):
        if c == -1:
            continue

        idxs = [i for i, cid in enumerate(point_cluster_ids) if cid == c]
        plt.scatter(
            x_pca[idxs, 0],
            x_pca[idxs, 1],
            color=cmap(c % cmap.N),
            s=45,
            alpha=0.85,
        )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("PCA (2D) of Predicate Embeddings")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_embedding_pca_3d(
    embedding_space: dict[str, object],
    mapping: dict[PRED_KEY, PRED_KEY],
    output_file: Path,
    cluster_size_threshold: int = 9
) -> None:
    x = embedding_space["embeddings"]
    labels = list(mapping.keys())

    # --- build clusters ---
    clusters: dict[PRED_KEY, list[PRED_KEY]] = {}
    for k, v in mapping.items():
        clusters.setdefault(v, []).append(k)

    cluster_id = {}
    cid = 0
    for rep, members in clusters.items():
        if len(members) > cluster_size_threshold:
            for m in members:
                cluster_id[m] = cid
            cid += 1

    point_cluster_ids = [
        cluster_id.get(lbl, -1)
        for lbl in labels
    ]

    # --- PCA ---
    pca = PCA(n_components=3, random_state=0)
    x_pca = pca.fit_transform(x)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # singletons (grey)
    singleton_idxs = [i for i, c in enumerate(point_cluster_ids) if c == -1]
    ax.scatter(
        x_pca[singleton_idxs, 0],
        x_pca[singleton_idxs, 1],
        x_pca[singleton_idxs, 2],
        c="lightgrey",
        s=25,
        alpha=0.5,
    )

    # clusters
    cmap = plt.cm.tab20
    for c in sorted(set(point_cluster_ids)):
        if c == -1:
            continue

        idxs = [i for i, cid in enumerate(point_cluster_ids) if cid == c]
        ax.scatter(
            x_pca[idxs, 0],
            x_pca[idxs, 1],
            x_pca[idxs, 2],
            color=cmap(c % cmap.N),
            s=45,
            alpha=0.85,
        )

    ax.set_title("PCA (3D) of Predicate Embeddings")

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_embedding_umap_2d(
    embedding_space: dict[str, object],
    mapping: dict[PRED_KEY, PRED_KEY],
    output_file: Path,
    min_cluster_size: int = 5
) -> None:
    x = embedding_space["embeddings"]

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.2,
        metric="cosine",
        random_state=0,
    )
    x_umap = reducer.fit_transform(x)

    # --- build clusters ---
    clusters: dict[PRED_KEY, list[int]] = {}
    for idx, k in enumerate(mapping.keys()):
        root = mapping[k]
        clusters.setdefault(root, []).append(idx)

    plt.figure(figsize=(8, 6))

    # singleton predicates → grey
    singleton_idx = [
        idx
        for root, idxs in clusters.items()
        if len(idxs) <= min_cluster_size
        for idx in idxs
    ]
    if singleton_idx:
        plt.scatter(
            x_umap[singleton_idx, 0],
            x_umap[singleton_idx, 1],
            s=35,
            c="lightgrey",
            alpha=0.6,
        )

    # multi-element clusters → colored
    cmap = plt.cm.tab20
    color_id = 0

    for idxs in clusters.values():
        if len(idxs) <= min_cluster_size:
            continue

        plt.scatter(
            x_umap[idxs, 0],
            x_umap[idxs, 1],
            s=45,
            color=cmap(color_id % cmap.N),
            alpha=0.85,
        )
        color_id += 1

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("UMAP projection of predicate embeddings")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_embedding_umap_3d(
        embedding_space: dict[str, object],
        mapping: dict[PRED_KEY, PRED_KEY],
        output_file: Path,
        min_cluster_size: int = 5
) -> None:
    x = embedding_space["embeddings"]

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.2,
        metric="cosine",
        random_state=0,
    )
    x_umap = reducer.fit_transform(x)

    # --- build clusters ---
    clusters: dict[PRED_KEY, list[int]] = {}
    for idx, k in enumerate(mapping.keys()):
        root = mapping[k]
        clusters.setdefault(root, []).append(idx)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # singleton predicates → grey
    singleton_idx = [
        idx
        for root, idxs in clusters.items()
        if len(idxs) <= min_cluster_size
        for idx in idxs
    ]
    if singleton_idx:
        ax.scatter(
            x_umap[singleton_idx, 0],
            x_umap[singleton_idx, 1],
            x_umap[singleton_idx, 2],
            s=35,
            c="lightgrey",
            alpha=0.6,
        )

    # multi-element clusters → colored
    cmap = plt.cm.tab20
    color_id = 0

    for idxs in clusters.values():
        if len(idxs) <= min_cluster_size:
            continue

        ax.scatter(
            x_umap[idxs, 0],
            x_umap[idxs, 1],
            x_umap[idxs, 2],
            s=45,
            color=cmap(color_id % cmap.N),
            alpha=0.85,
        )
        color_id += 1

    ax.set_title("3D UMAP of predicate embeddings")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)

    plot_embedding_pca_2d(
        embedding_space=embedding_predicate_space,
        mapping=map,
        output_file=PATH / 'predicate_embeddings_pca_2d.png'
    )

    plot_embedding_pca_3d(
        embedding_space=embedding_predicate_space,
        mapping=map,
        output_file=PATH / 'predicate_embeddings_pca_3d.png'
    )



if __name__ == "__main__":
    test_set = load_test_set()
    fol_formulae = test_set['FOL'].tolist()

    alphas = [0.1 * i for i in range(0, 11)]
    alpha = 0

    predicates = collect_predicates(fol_formulae)
    matrices = create_predicate_terms_matrices(fol_formulae)
    predicate_sentences = generate_predicate_embedding_sentences(matrices)
    arity_predicate_matrix = create_predicate_arity_matrix(list(predicates.keys()))

    embedding_predicate_space = build_embedding_space(
        list(predicate_sentences.values()),
        lambda x: TRANSFORMER.encode(x, normalize_embeddings=False)
    )

    semantic_predicate_matrix_score = build_predicate_similarity_matrix(
        embedding_predicate_space,
        predicate_sentences
    )

    matrix = arity_predicate_matrix * compute_similarity_scores(
        semantic_predicate_matrix_score,
        semantic_predicate_matrix_score,
        alpha=alpha
    )

    map = build_predicate_unification_map(
        sim_matrix=matrix,
        occurrences=predicates,
        threshold=0.9
    )

    plot_embedding_umap_2d(
        embedding_space=embedding_predicate_space,
        mapping=map,
        output_file=PATH / 'predicate_embeddings_umap_2d.png'
    )

    plot_embedding_umap_3d(
        embedding_space=embedding_predicate_space,
        mapping=map,
        output_file=PATH / 'predicate_embeddings_umap_3d.png'
    )





