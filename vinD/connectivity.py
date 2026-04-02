import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


NET_NAMES_ORDERED = [
    "visual", "somatomotor", "dorsal_attention",
    "ventral_attention", "limbic", "frontoparietal", "default_mode",
]


def compute_sliding_connectivity(network_ts, T, window_size=10):
    """Sliding-window Pearson connectivity between 7 Yeo networks.

    Returns
    -------
    connectivity_matrices : np.ndarray (n_windows, 7, 7)
    window_centers : np.ndarray (n_windows,)
    connectivity_pairs : dict[str, np.ndarray (n_windows,)]
    """
    if T < 2:
        raise ValueError("At least 2 timepoints are required to compute connectivity")

    effective_window = min(window_size, T)
    net_matrix = np.column_stack([network_ts[n] for n in NET_NAMES_ORDERED])

    n_windows = T - effective_window + 1
    connectivity_matrices = np.zeros((n_windows, 7, 7))
    window_centers = np.zeros(n_windows)

    for w in range(n_windows):
        start = w
        end = w + effective_window
        window_data = net_matrix[start:end, :]
        if window_data.std(axis=0).min() > 0:
            corr = np.corrcoef(window_data.T)
        else:
            corr = np.eye(7)
        connectivity_matrices[w] = corr
        window_centers[w] = (start + end) / 2

    connectivity_pairs = {
        "visual_limbic": connectivity_matrices[:, 0, 4],
        "visual_dorsal_attn": connectivity_matrices[:, 0, 2],
        "limbic_ventral_attn": connectivity_matrices[:, 4, 3],
        "frontoparietal_visual": connectivity_matrices[:, 5, 0],
        "frontoparietal_dmn": connectivity_matrices[:, 5, 6],
        "dmn_limbic": connectivity_matrices[:, 6, 4],
        "dmn_visual": connectivity_matrices[:, 6, 0],
    }

    return connectivity_matrices, window_centers, connectivity_pairs


def _upper_tri_features(matrices):
    """Extract upper triangle from each connectivity matrix."""
    n = matrices.shape[0]
    triu_idx = np.triu_indices(7, k=1)
    features = np.zeros((n, len(triu_idx[0])))
    for i in range(n):
        features[i] = matrices[i][triu_idx]
    return features


def cluster_brain_states(connectivity_matrices, window_centers, T, window_size=10):
    """KMeans clustering on FC upper-triangle features.

    Returns
    -------
    state_labels_per_sec : np.ndarray (T,)
    best_k : int
    state_labels_windows : np.ndarray (n_windows,)
    """
    n_windows = connectivity_matrices.shape[0]
    if n_windows == 0:
        raise ValueError("Connectivity input must contain at least one window")

    effective_window = min(window_size, T)
    fc_features = _upper_tri_features(connectivity_matrices)

    if n_windows == 1:
        state_labels_windows = np.zeros(1, dtype=int)
        state_labels_per_sec = np.zeros(T, dtype=int)
        return state_labels_per_sec, 1, state_labels_windows

    best_k = min(3, n_windows)
    best_score = -1
    max_k = min(6, n_windows - 1)
    k_range = range(2, max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels_k = km.fit_predict(fc_features)
        if len(np.unique(labels_k)) > 1:
            score = silhouette_score(fc_features, labels_k)
            if score > best_score:
                best_score = score
                best_k = k

    km_final = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    state_labels_windows = km_final.fit_predict(fc_features)

    # Map window-level states back to per-second states
    state_labels_per_sec = np.zeros(T, dtype=int)
    for w in range(n_windows):
        center_sec = int(window_centers[w])
        if center_sec < T:
            state_labels_per_sec[center_sec] = state_labels_windows[w]
    # Fill edges
    half_window = effective_window // 2
    if half_window > 0:
        state_labels_per_sec[:half_window] = state_labels_windows[0]
        state_labels_per_sec[-half_window:] = state_labels_windows[-1]

    return state_labels_per_sec, best_k, state_labels_windows
