import numpy as np

from vinD.connectivity import cluster_brain_states, compute_sliding_connectivity


def test_compute_sliding_connectivity_uses_short_inputs():
    timepoints = 4
    network_ts = {
        "visual": np.array([0.0, 1.0, 2.0, 3.0]),
        "somatomotor": np.array([3.0, 2.0, 1.0, 0.0]),
        "dorsal_attention": np.array([1.0, 1.0, 1.5, 2.0]),
        "ventral_attention": np.array([0.0, 0.5, 1.0, 1.5]),
        "limbic": np.array([2.0, 2.5, 3.0, 3.5]),
        "frontoparietal": np.array([1.0, 1.5, 1.0, 0.5]),
        "default_mode": np.array([0.5, 0.75, 1.0, 1.25]),
    }

    matrices, centers, pairs = compute_sliding_connectivity(network_ts, timepoints)

    assert matrices.shape == (1, 7, 7)
    assert centers.tolist() == [2.0]
    assert set(pairs) >= {"visual_limbic", "frontoparietal_dmn", "dmn_limbic"}


def test_cluster_brain_states_handles_single_window():
    matrices = np.eye(7, dtype=float)[None, :, :]
    centers = np.array([1.0])

    per_second, best_k, per_window = cluster_brain_states(matrices, centers, T=3)

    assert best_k == 1
    assert per_window.tolist() == [0]
    assert per_second.tolist() == [0, 0, 0]
