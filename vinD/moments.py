import numpy as np
from scipy import signal as scipy_signal
from scipy.ndimage import uniform_filter1d


def detect_critical_moments(metrics_normalized, viewer_states, T):
    """Identify key moments: hook, peaks, dips, emotional, novelty, transitions, momentum, ending.

    Returns
    -------
    moments : list[dict]
        Each dict has keys: time, start, end, type, score, severity, detail.
    """
    moments = []
    eng = metrics_normalized["total_engagement"].values

    # --- Hook (first 3 seconds) ---
    hook_score = eng[: min(3, T)].mean()
    moments.append({
        "time": "0:00-0:03",
        "start": 0,
        "end": min(3, T),
        "type": "hook",
        "score": round(hook_score, 1),
        "severity": (
            "strong" if hook_score > 65 else ("moderate" if hook_score > 40 else "weak")
        ),
        "detail": f"Hook strength: {hook_score:.0f}/100",
    })

    # --- Sustained engagement peaks (3+ seconds above 70th percentile) ---
    high_thresh = np.percentile(eng, 70)
    in_peak = False
    peak_start = 0
    for t in range(T):
        if eng[t] >= high_thresh and not in_peak:
            in_peak = True
            peak_start = t
        elif (eng[t] < high_thresh or t == T - 1) and in_peak:
            in_peak = False
            duration = t - peak_start
            if duration >= 3:
                peak_score = eng[peak_start:t].mean()
                dominant_states = viewer_states[peak_start:t].value_counts()
                dominant = (
                    dominant_states.index[0] if len(dominant_states) > 0 else "unknown"
                )
                moments.append({
                    "time": f"{peak_start // 60}:{peak_start % 60:02d}-{t // 60}:{t % 60:02d}",
                    "start": peak_start,
                    "end": t,
                    "type": "engagement_peak",
                    "score": round(peak_score, 1),
                    "severity": "strong" if peak_score > 80 else "moderate",
                    "detail": f"Sustained high engagement ({dominant.replace('_', ' ')})",
                })

    # --- Engagement dips (3+ seconds below 30th percentile) ---
    low_thresh = np.percentile(eng, 30)
    in_dip = False
    dip_start = 0
    for t in range(T):
        if eng[t] < low_thresh and not in_dip:
            in_dip = True
            dip_start = t
        elif (eng[t] >= low_thresh or t == T - 1) and in_dip:
            in_dip = False
            duration = t - dip_start
            if duration >= 3:
                dip_score = eng[dip_start:t].mean()
                dominant_states = viewer_states[dip_start:t].value_counts()
                dominant = (
                    dominant_states.index[0] if len(dominant_states) > 0 else "unknown"
                )
                moments.append({
                    "time": f"{dip_start // 60}:{dip_start % 60:02d}-{t // 60}:{t % 60:02d}",
                    "start": dip_start,
                    "end": t,
                    "type": "engagement_dip",
                    "score": round(dip_score, 1),
                    "severity": "concerning" if dip_score < 20 else "minor",
                    "detail": f"Viewer likely {dominant.replace('_', ' ')}",
                })

    # --- Arousal spikes (emotional peaks) ---
    arousal = metrics_normalized["arousal"].values
    peaks_idx, _ = scipy_signal.find_peaks(
        arousal, height=np.percentile(arousal, 80), distance=4
    )
    for p in peaks_idx:
        dominant = viewer_states.iloc[p]
        moments.append({
            "time": f"{p // 60}:{p % 60:02d}",
            "start": p,
            "end": p,
            "type": "emotional_peak",
            "score": round(arousal[p], 1),
            "severity": "strong" if arousal[p] > 80 else "moderate",
            "detail": f"Emotional arousal spike ({dominant.replace('_', ' ')})",
        })

    # --- Novelty spikes ---
    novelty = metrics_normalized["novelty"].values
    nov_peaks, _ = scipy_signal.find_peaks(
        novelty, height=np.percentile(novelty, 85), distance=3
    )
    for p in nov_peaks:
        moments.append({
            "time": f"{p // 60}:{p % 60:02d}",
            "start": p,
            "end": p,
            "type": "novelty_spike",
            "score": round(novelty[p], 1),
            "severity": "notable",
            "detail": "Visual change captured viewer attention",
        })

    # --- State transitions (notable ones) ---
    bad_transitions = {"mind_wandering", "checked_out", "confused"}
    for t in range(1, T):
        if viewer_states.iloc[t] != viewer_states.iloc[t - 1]:
            from_state = viewer_states.iloc[t - 1].replace("_", " ")
            to_state = viewer_states.iloc[t].replace("_", " ")
            if (
                viewer_states.iloc[t] in bad_transitions
                or viewer_states.iloc[t - 1] in bad_transitions
            ):
                moments.append({
                    "time": f"{t // 60}:{t % 60:02d}",
                    "start": t,
                    "end": t,
                    "type": "state_transition",
                    "score": round(eng[t], 1),
                    "severity": (
                        "warning"
                        if viewer_states.iloc[t] in bad_transitions
                        else "recovery"
                    ),
                    "detail": f"Transition: {from_state} → {to_state}",
                })

    # --- Momentum analysis (sustained negative trend) ---
    if T >= 10:
        smoothed = uniform_filter1d(eng.astype(float), size=5)
        gradient = np.gradient(smoothed)
        for t in range(5, T - 5):
            if all(gradient[t - 3 : t + 3] < -0.5):
                moments.append({
                    "time": f"{t // 60}:{t % 60:02d}",
                    "start": t,
                    "end": t,
                    "type": "momentum_loss",
                    "score": round(gradient[t] * 10, 1),
                    "severity": "warning",
                    "detail": "Engagement trending downward",
                })
                break

    # --- Ending strength (last 5 seconds) ---
    ending_eng = eng[-min(5, T) :].mean()
    end_start = max(0, T - 5)
    moments.append({
        "time": f"{end_start // 60}:{end_start % 60:02d}-{T // 60}:{T % 60:02d}",
        "start": end_start,
        "end": T,
        "type": "ending",
        "score": round(ending_eng, 1),
        "severity": (
            "strong"
            if ending_eng > 60
            else ("weak" if ending_eng < 35 else "moderate")
        ),
        "detail": f"Ending strength: {ending_eng:.0f}/100",
    })

    moments.sort(key=lambda x: x["start"])
    return moments
