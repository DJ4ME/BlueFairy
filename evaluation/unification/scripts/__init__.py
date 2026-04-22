from pathlib import Path
import pandas as pd
import numpy as np

TABLES_PATH = Path(__file__).parent.parent.resolve() / "tables"


def select_best_alpha_tau(
    df: pd.DataFrame,
    cluster_col="NumNonTrivialClusters",
    merge_col="NumMerges",
    lambda_smooth: float = 0.0,  # opzionale (non usato in opzione 1 base)
) -> tuple[float, float]:
    """
    Select best (alpha, tau) using a merge-aware scoring function:

        S = C * (1 - R)

    where:
        C = number of non-trivial clusters
        R = normalized number of merges

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
        ["Alpha", "Threshold", "NumMerges", "NumNonTrivialClusters"]
    cluster_col : str
        Column for cluster count.
    merge_col : str
        Column for number of merges.
    lambda_smooth : float
        (kept for compatibility; not used in this variant)

    Returns
    -------
    (alpha_best, tau_best)
    """

    best_candidates = []

    # global normalization for merges (important!)
    max_merges = df[merge_col].max()
    if max_merges == 0:
        max_merges = 1.0

    for alpha in sorted(df["Alpha"].unique()):
        sub = df[df["Alpha"] == alpha].sort_values("Threshold")

        clusters = sub[cluster_col].values.astype(float)
        merges = sub[merge_col].values.astype(float)
        taus = sub["Threshold"].values

        # ----------------------------
        # compute score S = C * (1 - R)
        # ----------------------------
        R = merges / max_merges
        score = clusters * (1.0 - R)

        # find maxima of score
        max_val = np.max(score)
        max_indices = np.where(score == max_val)[0]

        # ----------------------------
        # stability check (unchanged idea, but on score)
        # ----------------------------
        diffs = np.abs(np.diff(score))
        if len(diffs) == 0:
            continue

        iqr = np.percentile(diffs, 75) - np.percentile(diffs, 25)
        if iqr == 0:
            iqr = np.median(diffs) if np.median(diffs) > 0 else 1e-8

        stable_indices = []
        for idx in max_indices:
            left_diff = score[idx] - score[idx - 1] if idx > 0 else 0
            right_diff = score[idx + 1] - score[idx] if idx < len(score) - 1 else 0

            if abs(left_diff) <= iqr and abs(right_diff) <= iqr:
                stable_indices.append(idx)

        if not stable_indices:
            continue

        tau_star = np.median(taus[stable_indices])

        # recompute representative score at tau_star
        best_candidates.append((alpha, tau_star, max_val))

    if not best_candidates:
        raise ValueError("No stable maxima found for any alpha.")

    best_candidates = np.array(best_candidates, dtype=float)

    # optional: avoid syntactic-only regime
    # filtered = best_candidates[best_candidates[:, 0] < 1.0]

    best_idx = np.argmax(best_candidates[:, 2])
    alpha_best, tau_best = best_candidates[best_idx, 0], best_candidates[best_idx, 1]

    return float(alpha_best), float(tau_best)


if __name__ == "__main__":
    data = pd.read_csv(TABLES_PATH / "predicate_merging_data.csv")
    alpha_best, tau_best = select_best_alpha_tau(data)
    print(f"Selected best alpha: {alpha_best:.3f}, tau: {tau_best:.3f}")