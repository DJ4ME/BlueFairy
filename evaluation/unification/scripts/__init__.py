from pathlib import Path
import pandas as pd
import numpy as np


TABLES_PATH = Path(__file__).parent.parent.resolve() / 'tables'

def select_best_alpha_tau(df: pd.DataFrame, cluster_col="NumNonTrivialClusters") -> tuple[float, float]:
    """
    Select the best (alpha, tau) based on maximizing non-trivial clusters
    while ensuring stability around the peak using finite differences and IQR.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns ["Alpha", "Threshold", "NumMerges", "NumNonTrivialClusters"].
    cluster_col : str
        Column name to use for number of non-trivial clusters.

    Returns
    -------
    tuple[float, float]
        Selected (alpha_best, tau_best)
    """
    best_candidates = []

    # For each alpha, find the tau corresponding to maximum clusters
    for alpha in sorted(df["Alpha"].unique()):
        sub = df[df["Alpha"] == alpha].sort_values("Threshold")
        clusters = sub[cluster_col].values
        taus = sub["Threshold"].values

        # Find indices of maximum clusters
        max_val = clusters.max()
        max_indices = np.where(clusters == max_val)[0]

        # Check stability using finite differences and IQR of all cluster differences
        diffs = np.abs(np.diff(clusters))
        if len(diffs) == 0:
            continue
        #iqr = np.percentile(diffs, 75) - np.percentile(diffs, 25)
        # use the median instead of iqr
        iqr = np.median(diffs)

        stable_indices = []
        for idx in max_indices:
            # left and right differences
            left_diff = clusters[idx] - clusters[idx - 1] if idx > 0 else 0
            right_diff = clusters[idx + 1] - clusters[idx] if idx < len(clusters) - 1 else 0
            if abs(left_diff) <= iqr and abs(right_diff) <= iqr:
                stable_indices.append(idx)

        if not stable_indices:
            continue

        # Take the median tau of stable maxima
        tau_star = np.median(taus[stable_indices])
        best_candidates.append((alpha, tau_star, max_val))

    if not best_candidates:
        raise ValueError("No stable maxima found for any alpha.")

    # Convert to array for easier filtering
    best_candidates = np.array(best_candidates, dtype=float)

    # Filter out alpha == 1
    filtered = best_candidates[best_candidates[:, 0] < 1.0]

    # Select alpha with the highest cluster value among stable candidates
    best_idx = np.argmax(filtered[:, 2])
    alpha_best, tau_best = filtered[best_idx, 0], filtered[best_idx, 1]

    return float(alpha_best), float(tau_best)


if __name__ == "__main__":
    data = pd.read_csv(TABLES_PATH / 'predicate_merging_data.csv')
    alpha_best, tau_best = select_best_alpha_tau(data)
    print(f"Selected best alpha: {alpha_best:.3f}, tau: {tau_best:.3f}")