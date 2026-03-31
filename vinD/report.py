import numpy as np
import pandas as pd
from pathlib import Path

from vinD.connectivity import NET_NAMES_ORDERED


def generate_markdown_report(
    metrics_normalized,
    viewer_states,
    moments,
    state_labels_per_sec,
    best_k,
    connectivity_matrices,
    network_ts,
    coherence,
    avg_entropy,
    dwell_times,
    transition_info,
    memory_score,
    mem_peaks,
    has_subcortical,
    T,
    hrf_lag=5,
    segments=None,
):
    """Generate a full markdown report and return it as a string."""
    eng = metrics_normalized["total_engagement"].values
    lines = []

    lines.append("# Neural Focus Group Report\n")

    # --- Metadata ---
    lines.append("## Overview\n")
    lines.append(f"- **Video duration:** {T} seconds")
    lines.append(f"- **Cortical vertices:** fsaverage5 (20,484)")
    lines.append(f"- **Parcellation:** Schaefer 400 parcels, 7 Yeo networks")
    lines.append(f"- **HRF lag correction:** {hrf_lag}s")
    lines.append(f"- **Subcortical data:** {'Yes' if has_subcortical else 'No (using cortical proxies)'}")
    if segments is not None:
        lines.append(f"- **Segments provided:** {len(segments)}")
    lines.append("")

    # --- Overall Scores ---
    lines.append("## Overall Scores\n")
    lines.append("| Metric | Score (0-100) |")
    lines.append("|--------|:-------------:|")
    score_cols = [
        ("Total Engagement", "total_engagement"),
        ("Sensory Engagement", "sensory_engagement"),
        ("Emotional Engagement", "emotional_engagement"),
        ("Cognitive Engagement", "cognitive_engagement"),
        ("Social Connection", "social_engagement"),
        ("Narrative Resonance", "narrative_engagement"),
        ("Arousal", "arousal"),
        ("Mind-Wandering", "mind_wandering"),
    ]
    for label, col in score_cols:
        val = metrics_normalized[col].mean()
        lines.append(f"| {label} | {val:.1f} |")
    lines.append("")

    # --- Structure ---
    lines.append("## Structure Analysis\n")
    hook = eng[: min(3, T)].mean()
    ending = eng[-min(5, T) :].mean()
    retention_thresh = np.percentile(eng, 35)
    retained = (eng > retention_thresh).sum()
    retention_pct = retained / T * 100

    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|:-----:|")
    lines.append(f"| Hook Strength (0-3s) | {hook:.1f}/100 |")
    lines.append(f"| Ending Strength (last 5s) | {ending:.1f}/100 |")
    lines.append(f"| Predicted Retention | {retention_pct:.1f}% |")
    lines.append(f"| Narrative Coherence | {coherence:.0f}/100 |")
    lines.append(f"| Brain Dynamism (entropy) | {avg_entropy:.2f} (0=flat, 1=chaotic) |")
    lines.append("")

    # --- Entropy assessment ---
    if avg_entropy < 0.3:
        lines.append("> **Entropy assessment:** LOW — brain response is too flat, content is predictable\n")
    elif avg_entropy > 0.8:
        lines.append("> **Entropy assessment:** HIGH — brain response is chaotic, content may be incoherent\n")
    else:
        lines.append("> **Entropy assessment:** GOOD — dynamic but not chaotic\n")

    # --- Viewer State Breakdown ---
    lines.append("## Viewer State Breakdown\n")
    lines.append("| State | Time (s) | Percentage |")
    lines.append("|-------|:--------:|:----------:|")
    state_counts = viewer_states.value_counts()
    for state, count in state_counts.items():
        pct = count / T * 100
        lines.append(f"| {state.replace('_', ' ').title()} | {count}s | {pct:.1f}% |")
    lines.append("")

    # --- Dwell Times ---
    if dwell_times:
        lines.append("### Dwell Times\n")
        lines.append("| State | Avg Duration | Segments |")
        lines.append("|-------|:------------:|:--------:|")
        for state, times in sorted(dwell_times.items()):
            avg = np.mean(times)
            lines.append(f"| {state.replace('_', ' ').title()} | {avg:.1f}s | {len(times)} |")
        lines.append("")

    # --- State Transitions ---
    lines.append("## State Transitions\n")
    lines.append(f"- **Total transitions:** {transition_info['total_transitions']}")
    lines.append(f"- **Recovery transitions** (into engagement): {transition_info['good_transitions']}")
    lines.append(f"- **Loss transitions** (into disengagement): {transition_info['bad_transitions']}")
    lines.append(f"- **Deepening transitions** (between engagement types): {transition_info['deepening']}")
    lines.append("")

    # --- Brain Connectivity States ---
    lines.append("## Brain Connectivity States\n")
    lines.append(f"{best_k} distinct brain states detected via KMeans clustering on functional connectivity.\n")

    triu = np.triu_indices(7, k=1)
    pair_names = [
        f"{NET_NAMES_ORDERED[i][:4]}-{NET_NAMES_ORDERED[j][:4]}"
        for i, j in zip(triu[0], triu[1])
    ]

    for s in range(best_k):
        mask = np.zeros(len(connectivity_matrices), dtype=bool)
        # Find windows matching this state
        # state_labels_per_sec mapped back, but we need window-level labels
        # Just use per-sec for counting
        state_secs = np.where(state_labels_per_sec == s)[0]
        lines.append(f"### State {s} (~{len(state_secs)}s)\n")
        if len(state_secs) > 0:
            # Show time ranges
            ranges = []
            start = state_secs[0]
            for i in range(1, len(state_secs)):
                if state_secs[i] != state_secs[i - 1] + 1:
                    ranges.append(f"{start}-{state_secs[i - 1]}s")
                    start = state_secs[i]
            ranges.append(f"{start}-{state_secs[-1]}s")
            lines.append(f"**Time ranges:** {', '.join(ranges[:8])}")

            # Mean network activation
            lines.append("\n| Network | Mean Activation |")
            lines.append("|---------|:---------------:|")
            for net_name in NET_NAMES_ORDERED:
                mean_act = network_ts[net_name][state_secs].mean()
                lines.append(f"| {net_name.replace('_', ' ').title()} | {mean_act:.4f} |")
            lines.append("")

    # --- Memory Encoding ---
    lines.append("## Memory Encoding\n")
    lines.append(f"- **Average memorability:** {memory_score.mean():.1f}/100")
    lines.append(f"- **Peak memorable moments:** {len(mem_peaks)}")
    if len(mem_peaks) > 0:
        for p in mem_peaks:
            p_video = max(0, p - hrf_lag)
            lines.append(f"  - Neural sec {p} (video sec ~{p_video}): memorability={memory_score[p]:.0f}/100")
    else:
        lines.append("  - No strong memorable moments detected.")
        lines.append("  - Suggestion: add 2-3s of quieter content after emotional peaks.")
    lines.append("")

    # --- Critical Moments ---
    lines.append("## Critical Moments\n")
    lines.append(f"{len(moments)} moments detected.\n")
    lines.append("| Time | Type | Score | Severity | Detail |")
    lines.append("|------|------|:-----:|----------|--------|")
    for m in moments:
        # Show video-aligned time
        vs = max(0, m["start"] - hrf_lag)
        ve = max(0, m["end"] - hrf_lag)
        time_str = (
            f"{vs // 60}:{vs % 60:02d}"
            if vs == ve
            else f"{vs // 60}:{vs % 60:02d}-{ve // 60}:{ve % 60:02d}"
        )
        lines.append(
            f"| {time_str} | {m['type'].replace('_', ' ')} | {m['score']:.1f} | {m['severity']} | {m['detail']} |"
        )
    lines.append("")

    # --- Plots reference ---
    lines.append("## Generated Plots\n")
    plot_files = [
        ("timeline.png", "6-track detailed viewer experience timeline"),
        ("timeline_simple.png", "3-track simplified timeline"),
        ("connectivity.png", "Functional connectivity heatmaps per brain state"),
        ("connectivity_simple.png", "Simplified connectivity + brain integration score"),
        ("radar.png", "8-metric video neural profile radar chart"),
        ("coupling.png", "Dynamic network coupling over time"),
        ("coupling_simple.png", "3-question coupling narrative (See/Feel/Think)"),
        ("marketing_dashboard.png", "Composite marketing dashboard"),
        ("advanced_analysis.png", "Novelty, memorability, and temporal entropy"),
    ]
    for fname, desc in plot_files:
        lines.append(f"- **{fname}**: {desc}")
    lines.append("")

    return "\n".join(lines)


def save_report(report_text, output_dir):
    """Write the markdown report to output_dir/report.md."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(report_text, encoding="utf-8")
