from __future__ import annotations

import logging
from pathlib import Path

from vinD.advanced import (
    analyze_state_transitions,
    apply_hrf_correction,
    compute_memory_encoding,
    compute_temporal_entropy,
    extract_subcortical,
)
from vinD.connectivity import cluster_brain_states, compute_sliding_connectivity
from vinD.metrics import classify_viewer_states, compute_metrics
from vinD.moments import detect_critical_moments
from vinD.parcellation import (
    compute_network_timeseries,
    extract_key_regions,
    load_schaefer_atlas,
    parcellate_predictions,
)
from vinD.plots import save_all_plots
from vinD.report import generate_markdown_report, save_report


LOGGER = logging.getLogger("vinD.pipeline")


def _validate_predictions(preds) -> None:
    if not hasattr(preds, "shape"):
        raise TypeError("preds must be a NumPy-like 2D array")

    if len(preds.shape) != 2:
        raise ValueError(f"preds must be 2D with shape (time, vertices), got {preds.shape}")

    if preds.shape[0] < 2:
        raise ValueError("preds must contain at least 2 timepoints")

    if preds.shape[1] < 20484:
        raise ValueError(
            "preds must contain at least 20,484 cortical vertices from fsaverage5"
        )


def run_analysis(preds, segments=None, output_dir="results", hrf_lag=5):
    """Run the full neural focus group analysis pipeline."""
    _validate_predictions(preds)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    T = preds.shape[0]

    LOGGER.info("Analysing %ss of predictions across %s vertices", T, preds.shape[1])

    LOGGER.info("[1/8] Loading Schaefer 400-parcel atlas")
    vertex_labels, parcel_names = load_schaefer_atlas(n_parcels=400, n_networks=7)

    LOGGER.info("[2/8] Parcellating predictions")
    parcel_ts = parcellate_predictions(preds, vertex_labels, parcel_names)
    network_ts = compute_network_timeseries(parcel_ts, parcel_names)
    key_regions = extract_key_regions(parcel_ts, parcel_names)

    LOGGER.info("[3/8] Computing dynamic functional connectivity")
    connectivity_matrices, window_centers, connectivity_pairs = compute_sliding_connectivity(
        network_ts, T
    )
    state_labels_per_sec, best_k, state_labels_windows = cluster_brain_states(
        connectivity_matrices, window_centers, T
    )

    LOGGER.info("[4/8] Computing engagement metrics")
    metrics_raw, metrics_normalized = compute_metrics(network_ts, key_regions)
    viewer_states = classify_viewer_states(metrics_raw)
    metrics_normalized["viewer_state"] = viewer_states
    metrics_normalized["brain_state"] = state_labels_per_sec

    LOGGER.info("[5/8] Detecting critical moments")
    moments = detect_critical_moments(metrics_normalized, viewer_states, T)

    LOGGER.info("[6/8] Running advanced analyses")
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

    LOGGER.info("[7/8] Generating plots")
    eng = metrics_normalized["total_engagement"].values
    save_all_plots(
        plots_dir,
        metrics_normalized,
        viewer_states,
        moments,
        connectivity_matrices,
        window_centers,
        connectivity_pairs,
        state_labels_windows,
        best_k,
        eng,
        coherence,
        avg_entropy,
        memory_score,
        mem_peaks,
        T,
        hrf_lag,
    )

    LOGGER.info("[8/8] Writing report")
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

    LOGGER.info("Results saved to %s", output_dir.resolve())
    LOGGER.info("Generated report.md and %s plot files", len(list(plots_dir.glob("*.png"))))

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
