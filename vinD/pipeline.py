import numpy as np
from pathlib import Path

from vinD.parcellation import (
    load_schaefer_atlas,
    parcellate_predictions,
    compute_network_timeseries,
    extract_key_regions,
)
from vinD.connectivity import (
    compute_sliding_connectivity,
    cluster_brain_states,
)
from vinD.metrics import compute_metrics, classify_viewer_states
from vinD.moments import detect_critical_moments
from vinD.advanced import (
    compute_memory_encoding,
    extract_subcortical,
    analyze_state_transitions,
    compute_temporal_entropy,
    apply_hrf_correction,
)
from vinD.plots import save_all_plots
from vinD.report import generate_markdown_report, save_report


def run_analysis(preds, segments=None, output_dir="results", hrf_lag=5):
    """Run the full neural focus group analysis pipeline.

    Parameters
    ----------
    preds : np.ndarray, shape (T, n_vertices)
        TRIBE v2 predictions. T = number of timesteps (1 Hz), n_vertices >= 20484.
    segments : list, optional
        TRIBE v2 segment metadata. Stored in the report if provided.
    output_dir : str or Path
        Where to save report.md and plots/ folder.
    hrf_lag : int
        Hemodynamic lag in seconds for video-aligned timestamps.

    Returns
    -------
    results : dict
        All computed data for programmatic access.
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    T = preds.shape[0]

    print(f"vinD: Analysing {T}s of predictions ({preds.shape[1]} vertices)")

    # ── Step 1: Parcellation ──
    print("  [1/8] Loading Schaefer 400-parcel atlas...")
    vertex_labels, parcel_names = load_schaefer_atlas(n_parcels=400, n_networks=7)

    print("  [2/8] Parcellating predictions...")
    parcel_ts = parcellate_predictions(preds, vertex_labels, parcel_names)

    network_ts = compute_network_timeseries(parcel_ts, parcel_names)
    key_regions = extract_key_regions(parcel_ts, parcel_names)

    # ── Step 2: Dynamic connectivity ──
    print("  [3/8] Computing dynamic functional connectivity...")
    connectivity_matrices, window_centers, connectivity_pairs = (
        compute_sliding_connectivity(network_ts, T)
    )

    state_labels_per_sec, best_k, state_labels_windows = cluster_brain_states(
        connectivity_matrices, window_centers, T
    )

    # ── Step 3: Per-second metrics ──
    print("  [4/8] Computing engagement metrics...")
    metrics_raw, metrics_normalized = compute_metrics(network_ts, key_regions)
    viewer_states = classify_viewer_states(metrics_raw)
    metrics_normalized["viewer_state"] = viewer_states
    metrics_normalized["brain_state"] = state_labels_per_sec

    # ── Step 4: Critical moments ──
    print("  [5/8] Detecting critical moments...")
    moments = detect_critical_moments(metrics_normalized, viewer_states, T)

    # ── Step 5: Advanced analyses ──
    print("  [6/8] Running advanced analyses...")
    memory_score, mem_peaks = compute_memory_encoding(
        metrics_raw, network_ts, connectivity_pairs, window_centers, T
    )
    metrics_normalized["memorability"] = memory_score

    amygdala_arousal, hippocampal_memory, has_subcortical = extract_subcortical(preds, T)
    if has_subcortical:
        metrics_normalized["amygdala_arousal"] = amygdala_arousal
        metrics_normalized["hippocampal_memory"] = hippocampal_memory
    else:
        metrics_normalized["amygdala_arousal"] = metrics_normalized["arousal"]
        metrics_normalized["hippocampal_memory"] = metrics_normalized["memory_encoding"]

    transition_info, coherence, dwell_times = analyze_state_transitions(viewer_states, T)
    entropy_per_sec, avg_entropy = compute_temporal_entropy(metrics_normalized, T)
    metrics_normalized["temporal_entropy"] = (entropy_per_sec * 100).round(1)

    metrics_video_aligned, moments_video_aligned = apply_hrf_correction(
        metrics_normalized, moments, hrf_lag
    )

    # ── Step 6: Plots ──
    eng = metrics_normalized["total_engagement"].values
    print("  [7/8] Generating plots...")
    save_all_plots(
        plots_dir,
        metrics_normalized, viewer_states, moments,
        connectivity_matrices, window_centers, connectivity_pairs,
        state_labels_windows, best_k,
        eng, coherence, avg_entropy,
        memory_score, mem_peaks,
        T, hrf_lag,
    )

    # ── Step 7: Markdown report ──
    print("  [8/8] Writing report...")
    report_text = generate_markdown_report(
        metrics_normalized=metrics_normalized,
        viewer_states=viewer_states,
        moments=moments,
        state_labels_per_sec=state_labels_per_sec,
        best_k=best_k,
        connectivity_matrices=connectivity_matrices,
        network_ts=network_ts,
        coherence=coherence,
        avg_entropy=avg_entropy,
        dwell_times=dwell_times,
        transition_info=transition_info,
        memory_score=memory_score,
        mem_peaks=mem_peaks,
        has_subcortical=has_subcortical,
        T=T,
        hrf_lag=hrf_lag,
        segments=segments,
    )
    save_report(report_text, output_dir)

    print(f"\nDone! Results saved to {output_dir}/")
    print(f"  report.md + {len(list(plots_dir.glob('*.png')))} plots")

    return {
        "metrics_normalized": metrics_normalized,
        "metrics_raw": metrics_raw,
        "metrics_video_aligned": metrics_video_aligned,
        "viewer_states": viewer_states,
        "moments": moments,
        "moments_video_aligned": moments_video_aligned,
        "network_ts": network_ts,
        "key_regions": key_regions,
        "parcel_ts": parcel_ts,
        "connectivity_matrices": connectivity_matrices,
        "connectivity_pairs": connectivity_pairs,
        "window_centers": window_centers,
        "state_labels_per_sec": state_labels_per_sec,
        "best_k": best_k,
        "memory_score": memory_score,
        "mem_peaks": mem_peaks,
        "coherence": coherence,
        "avg_entropy": avg_entropy,
        "dwell_times": dwell_times,
        "transition_info": transition_info,
        "has_subcortical": has_subcortical,
        "report_text": report_text,
    }
