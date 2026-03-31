import math
import numpy as np
import pandas as pd
from collections import Counter
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from scipy.stats import zscore


def compute_memory_encoding(metrics_raw, network_ts, connectivity_pairs, window_centers, T):
    """Compute memory encoding score from emotional look-back × DMN derivative × DMN-emotion coupling.

    Returns
    -------
    memory_score : np.ndarray (T,)
        Normalized 0-100.
    mem_peaks : np.ndarray
        Indices of peak memorable moments.
    """
    emo_eng = metrics_raw["emotional_engagement"].values
    emo_smoothed = uniform_filter1d(emo_eng, size=5)

    dmn_raw = network_ts["default_mode"]
    dmn_derivative = np.gradient(dmn_raw)
    dmn_rising = np.clip(dmn_derivative, 0, None)

    # DMN-emotional coupling from connectivity matrices
    dmn_emo_coupling = np.zeros(T)
    for w in range(len(connectivity_pairs["dmn_limbic"])):
        center = int(window_centers[w])
        if center < T:
            dmn_emo_coupling[center] = connectivity_pairs["dmn_limbic"][w]

    # Interpolate gaps
    valid_idx = np.where(dmn_emo_coupling != 0)[0]
    if len(valid_idx) > 2:
        interp_func = interp1d(
            valid_idx, dmn_emo_coupling[valid_idx],
            fill_value="extrapolate", kind="linear",
        )
        dmn_emo_coupling = interp_func(np.arange(T))

    memory_score_raw = np.zeros(T)
    for t in range(T):
        lookback_start = max(0, t - 3)
        recent_emotion = emo_smoothed[lookback_start : t + 1].max() if t > 0 else 0
        dmn_rise = dmn_rising[t]
        coupling = max(0, dmn_emo_coupling[t])
        memory_score_raw[t] = recent_emotion * dmn_rise * coupling

    if memory_score_raw.max() > 0:
        memory_score = (memory_score_raw / memory_score_raw.max() * 100).round(1)
    else:
        memory_score = np.zeros(T)

    mem_peaks, _ = scipy_signal.find_peaks(memory_score, height=50, distance=5)
    return memory_score, mem_peaks


def extract_subcortical(preds, T):
    """Extract subcortical signals if preds has >20484 columns.

    Returns
    -------
    amygdala_arousal : np.ndarray (T,)
        Normalized 0-100.
    hippocampal_memory : np.ndarray (T,)
        Normalized 0-100.
    has_subcortical : bool
    """
    if preds.shape[1] <= 20484:
        return None, None, False

    subcort_preds = preds[:, 20484:]
    n_sub = subcort_preds.shape[1]

    structures = {
        "left_hippocampus": (3 * n_sub // 8, n_sub // 2 - n_sub // 16),
        "left_amygdala": (n_sub // 2 - n_sub // 16, n_sub // 2),
        "right_hippocampus": (n_sub // 2 + 3 * n_sub // 8, n_sub - n_sub // 16),
        "right_amygdala": (n_sub - n_sub // 16, n_sub),
    }

    subcort_ts = {}
    for name, (start, end) in structures.items():
        subcort_ts[name] = subcort_preds[:, start:end].mean(axis=1)

    amygdala_ts = (subcort_ts["left_amygdala"] + subcort_ts["right_amygdala"]) / 2
    hippocampus_ts = (subcort_ts["left_hippocampus"] + subcort_ts["right_hippocampus"]) / 2

    amygdala_z = zscore(amygdala_ts) if amygdala_ts.std() > 0 else amygdala_ts
    hippocampus_z = zscore(hippocampus_ts) if hippocampus_ts.std() > 0 else hippocampus_ts

    def _norm_0_100(arr):
        return np.clip(
            (arr - arr.min()) / (arr.max() - arr.min() + 1e-12) * 100, 0, 100
        ).round(1)

    return _norm_0_100(amygdala_z), _norm_0_100(hippocampus_z), True


def analyze_state_transitions(viewer_states, T):
    """Analyze transitions between viewer states.

    Returns
    -------
    info : dict
        Keys: total_transitions, good_transitions, bad_transitions, deepening.
    coherence : float
        Narrative coherence score 0-100.
    dwell_times : dict[str, list[int]]
    """
    positive_states = {
        "visually_absorbed", "emotionally_gripped",
        "personally_relating", "socially_connecting",
    }
    negative_states = {"mind_wandering", "checked_out", "confused"}

    transitions = []
    for t in range(1, T):
        if viewer_states.iloc[t] != viewer_states.iloc[t - 1]:
            transitions.append((viewer_states.iloc[t - 1], viewer_states.iloc[t]))

    good = sum(1 for f, t in transitions if t in positive_states and f not in positive_states)
    bad = sum(1 for f, t in transitions if t in negative_states and f not in negative_states)
    deepening = sum(1 for f, t in transitions if f in positive_states and t in positive_states and f != t)

    # Dwell times
    dwell_times = {}
    dwells = []
    current_state = viewer_states.iloc[0]
    current_start = 0
    for t_idx in range(1, T):
        if viewer_states.iloc[t_idx] != current_state:
            dwells.append((current_state, t_idx - current_start))
            current_state = viewer_states.iloc[t_idx]
            current_start = t_idx
    dwells.append((current_state, T - current_start))

    for state, duration in dwells:
        dwell_times.setdefault(state, []).append(duration)

    avg_dwell = np.mean([d for _, d in dwells]) if dwells else 0
    coherence = min(100, avg_dwell / T * 100 * 5)

    info = {
        "total_transitions": len(transitions),
        "good_transitions": good,
        "bad_transitions": bad,
        "deepening": deepening,
    }

    return info, coherence, dwell_times


def _permutation_entropy(signal, order=3, delay=1):
    """Compute normalized permutation entropy of a time series."""
    n = len(signal)
    permutations_count = Counter()
    for i in range(n - (order - 1) * delay):
        window = tuple(signal[i + j * delay] for j in range(order))
        sorted_idx = tuple(np.argsort(window))
        permutations_count[sorted_idx] += 1

    total = sum(permutations_count.values())
    probs = [count / total for count in permutations_count.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(math.factorial(order))
    return entropy / max_entropy if max_entropy > 0 else 0


def compute_temporal_entropy(metrics_normalized, T, window_size=10):
    """Compute per-second permutation entropy of total engagement.

    Returns
    -------
    entropy_per_sec : np.ndarray (T,)
        Raw entropy values (0-1).
    avg_entropy : float
    """
    engagement_vals = metrics_normalized["total_engagement"].values
    entropy_per_sec = np.zeros(T)

    for t in range(window_size, T):
        w = engagement_vals[t - window_size : t]
        entropy_per_sec[t] = _permutation_entropy(w, order=3)

    avg_entropy = entropy_per_sec[window_size:].mean() if T > window_size else 0.0
    return entropy_per_sec, avg_entropy


def apply_hrf_correction(metrics_normalized, moments, hrf_lag=5):
    """Shift metrics and moments backward to compensate for hemodynamic delay.

    Returns
    -------
    metrics_video_aligned : pd.DataFrame
    moments_video_aligned : list[dict]
    """
    metrics_video_aligned = metrics_normalized.copy()
    for col in metrics_video_aligned.columns:
        metrics_video_aligned[col] = metrics_video_aligned[col].shift(-hrf_lag)
    metrics_video_aligned = metrics_video_aligned.dropna().reset_index(drop=True)

    moments_video_aligned = []
    for m in moments:
        ms = m.copy()
        ms["start"] = max(0, m["start"] - hrf_lag)
        ms["end"] = max(0, m["end"] - hrf_lag)
        s, e = ms["start"], ms["end"]
        ms["time"] = (
            f"{s // 60}:{s % 60:02d}-{e // 60}:{e % 60:02d}"
            if s != e
            else f"{s // 60}:{s % 60:02d}"
        )
        if ms["start"] < len(metrics_video_aligned):
            moments_video_aligned.append(ms)

    return metrics_video_aligned, moments_video_aligned
