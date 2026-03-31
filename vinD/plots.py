import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy.ndimage import uniform_filter1d

from vinD.metrics import STATE_COLORS, FRIENDLY_NAMES
from vinD.connectivity import NET_NAMES_ORDERED


def _save_and_close(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# 1. Detailed 6-track timeline  (cell 33)
# ──────────────────────────────────────────────────────────────────────


def plot_detailed_timeline(metrics_normalized, viewer_states, moments, T, out_path):
    time = np.arange(T)
    fig, axes = plt.subplots(
        6, 1, figsize=(16, 14), sharex=True, gridspec_kw={"hspace": 0.08}
    )

    # Track 1: Total engagement
    ax = axes[0]
    ax.fill_between(
        time, 0, metrics_normalized["total_engagement"], alpha=0.3, color="#378ADD"
    )
    ax.plot(
        time, metrics_normalized["total_engagement"],
        color="#378ADD", linewidth=1.5, label="Total engagement",
    )
    ax.axhline(50, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_ylabel("Total\nEngagement", fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=8)

    # Track 2: Engagement breakdown
    ax = axes[1]
    for col, color, label in [
        ("sensory_engagement", "#378ADD", "Sensory"),
        ("emotional_engagement", "#E24B4A", "Emotional"),
        ("cognitive_engagement", "#7F77DD", "Cognitive"),
        ("narrative_engagement", "#1D9E75", "Narrative"),
    ]:
        ax.plot(time, metrics_normalized[col], color=color, linewidth=1, label=label, alpha=0.8)
    ax.axhline(50, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_ylabel("Engagement\nBreakdown", fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=7, ncol=4)

    # Track 3: Arousal + Valence
    ax = axes[2]
    ax.plot(time, metrics_normalized["arousal"], color="#E24B4A", linewidth=1.2, label="Arousal")
    ax.plot(time, metrics_normalized["valence_proxy"], color="#1D9E75", linewidth=1.2, label="Valence proxy")
    ax.axhline(50, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_ylabel("Arousal &\nValence", fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=8)

    # Track 4: Key regions
    ax = axes[3]
    for col, color, label in [
        ("face_response", "#D4537E", "Face"),
        ("language_response", "#534AB7", "Language"),
        ("memory_encoding", "#0F6E56", "Memory"),
        ("reward_signal", "#BA7517", "Reward"),
    ]:
        ax.plot(time, metrics_normalized[col], color=color, linewidth=1, label=label, alpha=0.8)
    ax.axhline(50, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_ylabel("Key\nRegions", fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=7, ncol=4)

    # Track 5: Mind-wandering + Novelty
    ax = axes[4]
    ax.fill_between(
        time, 0, metrics_normalized["mind_wandering"],
        alpha=0.4, color="#888780", label="Mind-wandering",
    )
    ax.plot(
        time, metrics_normalized["novelty"], color="#EF9F27",
        linewidth=1.2, label="Novelty", alpha=0.9,
    )
    ax.set_ylabel("Wandering\n& Novelty", fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=8)

    # Track 6: State ribbon
    ax = axes[5]
    for t_sec in range(T):
        state = viewer_states.iloc[t_sec]
        color = STATE_COLORS.get(state, "#B4B2A9")
        ax.axvspan(t_sec, t_sec + 1, color=color, alpha=0.8)
    ax.set_ylabel("Viewer\nState", fontsize=10)
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)", fontsize=11)
    used_states = viewer_states.unique()
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, fc=STATE_COLORS.get(s, "#B4B2A9"), alpha=0.8)
        for s in used_states
    ]
    ax.legend(
        legend_patches,
        [s.replace("_", " ") for s in used_states],
        loc="upper center", fontsize=7, ncol=len(used_states),
        bbox_to_anchor=(0.5, -0.3),
    )

    # Mark critical moments on top track
    for m in moments:
        if m["type"] in ["emotional_peak", "novelty_spike"]:
            axes[0].axvline(m["start"], color="#E24B4A", linewidth=0.8, alpha=0.5, linestyle=":")
        elif m["type"] in ["engagement_dip", "momentum_loss"]:
            axes[0].axvline(m["start"], color="#888780", linewidth=0.8, alpha=0.5, linestyle=":")

    fig.suptitle(
        "Neural Focus Group — Comprehensive Viewer Experience Timeline",
        fontsize=14, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    _save_and_close(fig, out_path)


# ──────────────────────────────────────────────────────────────────────
# 2. Simplified 3-track timeline  (cell 34)
# ──────────────────────────────────────────────────────────────────────


def plot_simple_timeline(metrics_normalized, viewer_states, moments, T, out_path):
    fig, axes = plt.subplots(
        3, 1, figsize=(16, 9), sharex=True,
        gridspec_kw={"hspace": 0.08, "height_ratios": [3, 2, 0.6]},
    )
    time = np.arange(T)
    eng = metrics_normalized["total_engagement"].values

    # Track 1: Attention with colored fill
    ax = axes[0]
    ax.plot(time, eng, color="#185FA5", linewidth=2.5, zorder=3)
    for t in range(T - 1):
        state = viewer_states.iloc[t]
        color = STATE_COLORS.get(state, "#B4B2A9")
        ax.fill_between([t, t + 1], 0, [eng[t], eng[t + 1]], color=color, alpha=0.35)

    for m in moments:
        mt = m["start"]
        if mt >= T:
            continue
        y = min(95, eng[min(mt, T - 1)] + 6)
        if m["type"] == "emotional_peak":
            ax.annotate("\u2665", (mt, y), fontsize=16, ha="center", color="#E24B4A", zorder=5)
        elif m["type"] == "engagement_dip" and m["severity"] == "concerning":
            mid = m["start"] + (m["end"] - m["start"]) // 2
            ax.annotate("\u2B07", (mid, eng[min(mt, T - 1)] + 6), fontsize=12, ha="center", color="#888780", zorder=5)
        elif m["type"] == "novelty_spike":
            ax.annotate("\u2605", (mt, y), fontsize=13, ha="center", color="#EF9F27", zorder=5)
        elif m["type"] == "momentum_loss":
            ax.annotate("\u2198", (mt, y), fontsize=14, ha="center", color="#E24B4A", zorder=5)

    ax.axhline(50, color="gray", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.text(T + 0.5, 50, "avg", fontsize=8, color="gray", va="center")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Viewer Attention", fontsize=12, fontweight="bold")
    ax.set_title("How engaged is the viewer? (color = type of engagement)", fontsize=13, fontweight="bold")

    used = viewer_states.unique()
    patches = [plt.Rectangle((0, 0), 1, 1, fc=STATE_COLORS.get(s, "#B4B2A9"), alpha=0.5) for s in used]
    labels = [FRIENDLY_NAMES.get(s, s) for s in used]
    patches += [
        plt.Line2D([0], [0], marker="$\u2665$", color="#E24B4A", linestyle="None", markersize=10),
        plt.Line2D([0], [0], marker="$\u2605$", color="#EF9F27", linestyle="None", markersize=10),
        plt.Line2D([0], [0], marker="$\u2198$", color="#E24B4A", linestyle="None", markersize=10),
    ]
    labels += ["Emotional peak", "Surprise moment", "Losing momentum"]
    ax.legend(patches, labels, loc="upper right", fontsize=7, ncol=3)

    # Track 2: Energy level
    ax = axes[1]
    arousal_n = metrics_normalized["arousal"].values
    novelty_n = metrics_normalized["novelty"].values
    energy = np.maximum(arousal_n, novelty_n)
    energy_smooth = uniform_filter1d(energy.astype(float), size=3)

    ax.fill_between(time, 0, energy_smooth, color="#EF9F27", alpha=0.2)
    ax.plot(time, energy_smooth, color="#EF9F27", linewidth=2, label="Energy (excitement level)")
    face_n = metrics_normalized["face_response"].values
    ax.plot(time, face_n, color="#D4537E", linewidth=1.5, alpha=0.7, label="Face on screen")
    ax.axhline(50, color="gray", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Energy Level", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)

    # Track 3: State ribbon
    ax = axes[2]
    for t_sec in range(T):
        state = viewer_states.iloc[t_sec]
        color = STATE_COLORS.get(state, "#B4B2A9")
        ax.axvspan(t_sec, t_sec + 1, color=color, alpha=0.85)
    ax.set_ylabel("State", fontsize=10, fontweight="bold")
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_xlim(0, T)

    plt.tight_layout()
    _save_and_close(fig, out_path)


# ──────────────────────────────────────────────────────────────────────
# 3. Connectivity heatmaps per brain state  (cell 35)
# ──────────────────────────────────────────────────────────────────────


def plot_connectivity_states(connectivity_matrices, state_labels_windows, best_k, T, out_path):
    n_cols = min(best_k + 1, 5)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    short_names = ["Vis", "SomMot", "DorsAtt", "VentAtt", "Limb", "FrontP", "DMN"]
    mask = np.triu(np.ones((7, 7), dtype=bool), k=1)

    ax = axes[0]
    mean_fc = connectivity_matrices.mean(axis=0)
    sns.heatmap(
        mean_fc, mask=~mask, ax=ax, cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.8,
        xticklabels=short_names, yticklabels=short_names,
        square=True, cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Overall\navg connectivity", fontsize=10)
    ax.tick_params(labelsize=7)

    for s in range(min(best_k, len(axes) - 1)):
        ax = axes[s + 1]
        state_mask = state_labels_windows == s
        if state_mask.sum() > 0:
            state_fc = connectivity_matrices[state_mask].mean(axis=0)
            sns.heatmap(
                state_fc, mask=~mask, ax=ax, cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.8,
                xticklabels=short_names, yticklabels=short_names,
                square=True, cbar_kws={"shrink": 0.7},
            )
            duration = state_mask.sum()
            ax.set_title(f"State {s}\n({duration}s)", fontsize=10)
        ax.tick_params(labelsize=7)

    fig.suptitle("Dynamic Functional Connectivity by Brain State", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_and_close(fig, out_path)


# ──────────────────────────────────────────────────────────────────────
# 4. Simplified connectivity + integration score  (cell 36)
# ──────────────────────────────────────────────────────────────────────


def plot_connectivity_simple(connectivity_matrices, window_centers, state_labels_windows, T, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    short_names = ["Visual", "Body", "Focus", "Salience", "Emotion", "Thinking", "Daydream"]

    # Left: heatmap
    ax = axes[0]
    mean_fc = connectivity_matrices.mean(axis=0)
    mask = np.zeros_like(mean_fc, dtype=bool)
    mask[np.triu_indices_from(mask, k=0)] = True
    sns.heatmap(
        mean_fc, mask=mask, ax=ax, cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.8,
        xticklabels=short_names, yticklabels=short_names,
        square=True, cbar_kws={"shrink": 0.8, "label": "Coupling strength"},
        annot=True, fmt=".1f", annot_kws={"fontsize": 8},
    )
    ax.set_title("How brain networks talk to each other\n(overall average)", fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=9)

    # Right: integration score
    ax = axes[1]
    integration_score = np.zeros(len(connectivity_matrices))
    for w in range(len(connectivity_matrices)):
        triu_vals = connectivity_matrices[w][np.triu_indices(7, k=1)]
        integration_score[w] = triu_vals.mean()

    integration_norm = np.clip(
        (integration_score - integration_score.min())
        / (integration_score.max() - integration_score.min() + 1e-12) * 100,
        0, 100,
    )
    ax.fill_between(window_centers, 0, integration_norm, color="#534AB7", alpha=0.2)
    ax.plot(window_centers, integration_norm, color="#534AB7", linewidth=2)

    for w in range(len(state_labels_windows) - 1):
        ax.axvspan(
            window_centers[w], window_centers[w + 1],
            color=f"C{state_labels_windows[w]}", alpha=0.08,
        )

    ax.axhline(50, color="gray", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, T)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Brain Integration", fontsize=11, fontweight="bold")
    ax.set_title(
        "How unified is brain processing?\n(high = all systems aligned, low = fragmented)",
        fontsize=11, fontweight="bold",
    )

    plt.tight_layout()
    _save_and_close(fig, out_path)


# ──────────────────────────────────────────────────────────────────────
# 5. Radar chart  (cell 37)
# ──────────────────────────────────────────────────────────────────────


def plot_radar(metrics_normalized, out_path):
    categories = [
        "Sensory\nEngagement", "Emotional\nEngagement", "Cognitive\nEngagement",
        "Social\nConnection", "Narrative\nResonance", "Memory\nEncoding",
        "Arousal", "Novelty",
    ]
    values = [
        metrics_normalized["sensory_engagement"].mean(),
        metrics_normalized["emotional_engagement"].mean(),
        metrics_normalized["cognitive_engagement"].mean(),
        metrics_normalized["social_engagement"].mean(),
        metrics_normalized["narrative_engagement"].mean(),
        metrics_normalized["memory_encoding"].mean(),
        metrics_normalized["arousal"].mean(),
        metrics_normalized["novelty"].mean(),
    ]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles_plot = angles + [angles[0]]

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.fill(angles_plot, values_plot, color="#378ADD", alpha=0.15)
    ax.plot(angles_plot, values_plot, color="#378ADD", linewidth=2)
    ax.scatter(angles, values, color="#378ADD", s=60, zorder=5)

    for angle, val, cat in zip(angles, values, categories):
        ax.annotate(
            f"{val:.0f}", xy=(angle, val), fontsize=10, ha="center", va="bottom",
            fontweight="bold", color="#185FA5",
        )

    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75])
    ax.set_yticklabels(["25", "50", "75"], fontsize=8, color="gray")
    ax.set_title("Video Neural Profile", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    _save_and_close(fig, out_path)


# ──────────────────────────────────────────────────────────────────────
# 6. Dynamic network coupling  (cell 38)
# ──────────────────────────────────────────────────────────────────────


def plot_coupling(connectivity_pairs, window_centers, T, out_path):
    fig, ax = plt.subplots(figsize=(16, 5))
    pair_colors = {
        "visual_limbic": ("#E24B4A", "Visual-Emotional coupling"),
        "visual_dorsal_attn": ("#378ADD", "Visual-Attention coupling"),
        "frontoparietal_dmn": ("#7F77DD", "Executive-DMN coupling"),
        "dmn_limbic": ("#1D9E75", "DMN-Emotional coupling"),
    }

    for key, (color, label) in pair_colors.items():
        vals = connectivity_pairs[key]
        smoothed = uniform_filter1d(vals, size=3)
        ax.plot(
            window_centers[: len(smoothed)], smoothed,
            color=color, linewidth=1.5, label=label, alpha=0.85,
        )

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Correlation (r)", fontsize=11)
    ax.set_title("Dynamic Network Coupling Over Time", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.set_xlim(0, T)
    plt.tight_layout()
    _save_and_close(fig, out_path)


# ──────────────────────────────────────────────────────────────────────
# 7. Simplified coupling — 3 narrative questions  (cell 39)
# ──────────────────────────────────────────────────────────────────────


def plot_coupling_simple(connectivity_pairs, window_centers, T, out_path):
    fig, axes = plt.subplots(
        3, 1, figsize=(16, 8), sharex=True, gridspec_kw={"hspace": 0.15}
    )
    wc = window_centers
    color_pos = "#1D9E75"
    color_neg = "#E24B4A"

    pairs_config = [
        (
            "visual_limbic", axes[0], "See \u2192 Feel",
            '"When the viewer sees something, do they feel it?"',
            ("YES\n(good)", color_pos), ("NO\n(flat)", color_neg),
        ),
        (
            "dmn_limbic", axes[1], "Feel \u2192 Relate",
            '"When the viewer feels something, do they connect it to themselves?"',
            ("YES\n(deep)", color_pos), ("NO\n(surface)", color_neg),
        ),
        (
            "frontoparietal_visual", axes[2], "Think \u2192 See",
            '"Is the viewer actively thinking about what they see, or passively watching?"',
            ("Analyzing\n(active)", "#7F77DD"), ("Passive\n(watching)", "#EF9F27"),
        ),
    ]

    for key, ax, ylabel, title, pos_label, neg_label in pairs_config:
        vals = connectivity_pairs[key]
        vals_smooth = uniform_filter1d(vals, size=3)
        p_color = pos_label[1]
        n_color = neg_label[1]
        ax.fill_between(wc[: len(vals_smooth)], 0, vals_smooth, where=vals_smooth >= 0, color=p_color, alpha=0.25)
        ax.fill_between(wc[: len(vals_smooth)], 0, vals_smooth, where=vals_smooth < 0, color=n_color, alpha=0.25)
        ax.plot(wc[: len(vals_smooth)], vals_smooth, color="#185FA5", linewidth=2)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_ylim(-1, 1)
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.text(T + 1, 0.5, pos_label[0], fontsize=8, color=p_color, va="center")
        ax.text(T + 1, -0.5, neg_label[0], fontsize=8, color=n_color, va="center")

    axes[2].set_xlabel("Time (seconds)", fontsize=11)
    plt.tight_layout()
    _save_and_close(fig, out_path)


# ──────────────────────────────────────────────────────────────────────
# 8. Marketing dashboard  (cell 45)
# ──────────────────────────────────────────────────────────────────────


def plot_marketing_dashboard(
    metrics_normalized, viewer_states, moments,
    eng, coherence, avg_entropy, T, hrf_lag, out_path,
):
    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(
        4, 3, hspace=0.35, wspace=0.3, height_ratios=[2, 1.5, 1, 1.2]
    )
    time = np.arange(T)
    eng_total = metrics_normalized["total_engagement"].values

    # === Top row: Engagement timeline ===
    ax_timeline = fig.add_subplot(gs[0, :])
    ax_timeline.plot(time, eng_total, color="#185FA5", linewidth=2, zorder=3)
    for t in range(T - 1):
        state = viewer_states.iloc[t]
        color = STATE_COLORS.get(state, "#B4B2A9")
        ax_timeline.fill_between(
            [t, t + 1], 0, [eng_total[t], eng_total[t + 1]], color=color, alpha=0.4,
        )

    for m in moments:
        mt = m["start"]
        if mt >= T:
            continue
        y_val = min(95, eng_total[min(mt, T - 1)] + 5)
        if m["type"] == "emotional_peak":
            ax_timeline.annotate("\u2665", (mt, y_val), fontsize=14, ha="center", color="#E24B4A")
        elif m["type"] == "engagement_dip" and m["severity"] == "concerning":
            ax_timeline.annotate("\u2193", (mt, y_val), fontsize=14, ha="center", color="#888780", fontweight="bold")
        elif m["type"] == "novelty_spike":
            ax_timeline.annotate("\u2605", (mt, y_val), fontsize=12, ha="center", color="#EF9F27")

    ax_timeline.set_ylim(0, 100)
    ax_timeline.set_xlim(0, T)
    ax_timeline.set_ylabel("Viewer Attention", fontsize=12, fontweight="bold")
    ax_timeline.set_xlabel("Time (seconds)", fontsize=11)
    ax_timeline.set_title("Viewer Attention Over Time", fontsize=14, fontweight="bold")

    used_states = viewer_states.unique()
    patches = [
        plt.Rectangle((0, 0), 1, 1, fc=STATE_COLORS.get(s, "#B4B2A9"), alpha=0.6)
        for s in used_states
    ]
    ax_timeline.legend(
        patches, [FRIENDLY_NAMES.get(s, s) for s in used_states],
        loc="upper left", fontsize=8, ncol=min(4, len(used_states)),
    )

    # === Middle left: Radar ===
    ax_radar = fig.add_subplot(gs[1, 0], polar=True)
    friendly_cats = [
        "Eye-\ncatching", "Heart-\ntouching", "Brain-\nstimulating",
        "Connecting", "Personal", "Memorable", "Exciting", "Surprising",
    ]
    radar_values = [
        metrics_normalized["sensory_engagement"].mean(),
        metrics_normalized["emotional_engagement"].mean(),
        metrics_normalized["cognitive_engagement"].mean(),
        metrics_normalized["social_engagement"].mean(),
        metrics_normalized["narrative_engagement"].mean(),
        metrics_normalized.get("hippocampal_memory", metrics_normalized["memory_encoding"]).mean(),
        metrics_normalized["arousal"].mean(),
        metrics_normalized["novelty"].mean(),
    ]
    angles = np.linspace(0, 2 * np.pi, len(friendly_cats), endpoint=False).tolist()
    rv_plot = radar_values + [radar_values[0]]
    a_plot = angles + [angles[0]]
    ax_radar.fill(a_plot, rv_plot, color="#378ADD", alpha=0.15)
    ax_radar.plot(a_plot, rv_plot, color="#378ADD", linewidth=2)
    ax_radar.scatter(angles, radar_values, color="#378ADD", s=50, zorder=5)
    for a, v in zip(angles, radar_values):
        ax_radar.annotate(f"{v:.0f}", xy=(a, v), fontsize=9, ha="center", va="bottom", fontweight="bold", color="#185FA5")
    ax_radar.set_xticks(angles)
    ax_radar.set_xticklabels(friendly_cats, fontsize=8)
    ax_radar.set_ylim(0, 100)
    ax_radar.set_yticks([25, 50, 75])
    ax_radar.set_yticklabels(["", "", ""], fontsize=7)
    ax_radar.set_title("Video Profile", fontsize=12, fontweight="bold", pad=15)

    # === Middle center: Score cards ===
    ax_scores = fig.add_subplot(gs[1, 1])
    ax_scores.axis("off")
    retention = (eng > np.percentile(eng, 35)).sum() / T * 100
    score_data = [
        ("Hook Strength", eng[: min(3, T)].mean(), "#E24B4A" if eng[: min(3, T)].mean() < 30 else "#1D9E75"),
        ("Predicted Retention", retention, "#378ADD"),
        ("Ending Strength", eng[-min(5, T) :].mean(), "#7F77DD"),
        ("Narrative Coherence", coherence, "#BA7517"),
        ("Brain Dynamism", avg_entropy * 100, "#534AB7"),
    ]
    for i, (label, value, color) in enumerate(score_data):
        y = 0.9 - i * 0.18
        ax_scores.text(0.05, y, label, fontsize=11, va="center", transform=ax_scores.transAxes)
        ax_scores.text(
            0.85, y, f"{value:.0f}", fontsize=18, fontweight="bold",
            va="center", ha="right", color=color, transform=ax_scores.transAxes,
        )
        bar_width = value / 100 * 0.55
        ax_scores.barh(y, bar_width, height=0.06, left=0.3, color=color, alpha=0.2, transform=ax_scores.transAxes)
    ax_scores.set_title("Key Scores", fontsize=12, fontweight="bold")

    # === Middle right: Donut ===
    ax_donut = fig.add_subplot(gs[1, 2])
    state_counts = viewer_states.value_counts()
    colors_donut = [STATE_COLORS.get(s, "#B4B2A9") for s in state_counts.index]
    labels_donut = [FRIENDLY_NAMES.get(s, s) for s in state_counts.index]
    wedges, texts, autotexts = ax_donut.pie(
        state_counts.values, labels=labels_donut, colors=colors_donut,
        autopct="%1.0f%%", pctdistance=0.75, startangle=90,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
    )
    for t in texts:
        t.set_fontsize(8)
    for t in autotexts:
        t.set_fontsize(8)
        t.set_fontweight("bold")
    ax_donut.set_title("How the viewer spent their time", fontsize=12, fontweight="bold")

    # === Bottom: Key moments table ===
    ax_moments = fig.add_subplot(gs[2:, :])
    ax_moments.axis("off")
    ax_moments.set_title("Key Moments", fontsize=12, fontweight="bold", loc="left", pad=10)

    icon_map = {
        "hook": "\u25C6", "engagement_peak": "\u25B2", "engagement_dip": "\u25BC",
        "emotional_peak": "\u2665", "novelty_spike": "\u2605", "state_transition": "\u2192",
        "ending": "\u25C6", "momentum_loss": "\u2198",
    }
    severity_colors = {
        "strong": "#1D9E75", "moderate": "#378ADD", "notable": "#EF9F27",
        "weak": "#E24B4A", "concerning": "#E24B4A", "warning": "#EF9F27",
        "recovery": "#1D9E75",
    }

    important_types = {"hook", "engagement_peak", "engagement_dip", "emotional_peak", "momentum_loss", "ending"}
    key_moments = [m for m in moments if m["type"] in important_types][:10]

    y_start = 0.92
    ax_moments.text(0.95, y_start + 0.06, "Score", fontsize=9, va="center", ha="right", fontweight="bold", color="gray", transform=ax_moments.transAxes)
    ax_moments.text(0.05, y_start + 0.06, "When", fontsize=9, va="center", fontweight="bold", color="gray", transform=ax_moments.transAxes)
    ax_moments.text(0.18, y_start + 0.06, "What happened", fontsize=9, va="center", fontweight="bold", color="gray", transform=ax_moments.transAxes)

    for i, m in enumerate(key_moments):
        y = y_start - i * 0.09
        if y < 0:
            break
        icon = icon_map.get(m["type"], "\u2022")
        color = severity_colors.get(m["severity"], "#888780")
        video_start = max(0, m["start"] - hrf_lag)
        video_end = max(0, m["end"] - hrf_lag)
        time_str = (
            f"{video_start // 60}:{video_start % 60:02d}"
            if video_start == video_end
            else f"{video_start // 60}:{video_start % 60:02d}-{video_end // 60}:{video_end % 60:02d}"
        )
        ax_moments.text(0.02, y, icon, fontsize=14, va="center", color=color, transform=ax_moments.transAxes)
        ax_moments.text(0.05, y, time_str, fontsize=10, va="center", fontfamily="monospace", transform=ax_moments.transAxes)
        ax_moments.text(0.18, y, m["detail"], fontsize=10, va="center", transform=ax_moments.transAxes, color="#333333")
        badge_color = "#1D9E75" if m["score"] > 60 else ("#EF9F27" if m["score"] > 35 else "#E24B4A")
        ax_moments.text(0.95, y, f"{m['score']:.0f}", fontsize=11, va="center", ha="right", fontweight="bold", color=badge_color, transform=ax_moments.transAxes)

    fig.suptitle("Neural Focus Group Report", fontsize=16, fontweight="bold", y=0.98)
    _save_and_close(fig, out_path)


# ──────────────────────────────────────────────────────────────────────
# 9. Advanced analysis  (cell 46)
# ──────────────────────────────────────────────────────────────────────


def plot_advanced_analysis(metrics_normalized, memory_score, mem_peaks, T, out_path):
    fig, axes = plt.subplots(
        3, 1, figsize=(16, 8), sharex=True, gridspec_kw={"hspace": 0.15}
    )
    time = np.arange(T)

    # Track 1: Novelty
    ax = axes[0]
    ax.plot(time, metrics_normalized["novelty"], color="#EF9F27", linewidth=1.5, label="Visual change (editing)", alpha=0.8)
    ax.axhline(50, color="gray", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.set_ylabel("Surprise\nLevel", fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Visual Novelty", fontsize=12, fontweight="bold")

    # Track 2: Memory encoding
    ax = axes[1]
    ax.fill_between(time, 0, metrics_normalized["memorability"], color="#0F6E56", alpha=0.3)
    ax.plot(time, metrics_normalized["memorability"], color="#0F6E56", linewidth=1.5, label="Memorability score")
    if len(mem_peaks) > 0:
        ax.scatter(mem_peaks, memory_score[mem_peaks], color="#0F6E56", s=80, zorder=5, marker="*", label="Peak memorable moments")
    ax.axhline(50, color="gray", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.set_ylabel("Will they\nremember?", fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title('Memory Encoding Score \u2014 "Will the viewer remember this?"', fontsize=12, fontweight="bold")

    # Track 3: Temporal entropy
    ax = axes[2]
    ax.fill_between(time, 0, metrics_normalized["temporal_entropy"], color="#534AB7", alpha=0.2)
    ax.plot(time, metrics_normalized["temporal_entropy"], color="#534AB7", linewidth=1.5, label="Brain dynamism")
    ax.axhline(30, color="#E24B4A", linewidth=0.8, linestyle=":", alpha=0.5, label="Too flat (boring)")
    ax.axhline(80, color="#EF9F27", linewidth=0.8, linestyle=":", alpha=0.5, label="Too chaotic")
    ax.set_ylabel("Brain\nDynamism", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Temporal Entropy \u2014 Is the brain actively processing or flat-lining?", fontsize=12, fontweight="bold")

    plt.tight_layout()
    _save_and_close(fig, out_path)


# ──────────────────────────────────────────────────────────────────────
# Convenience: save all plots
# ──────────────────────────────────────────────────────────────────────


def save_all_plots(
    plots_dir,
    metrics_normalized, viewer_states, moments,
    connectivity_matrices, window_centers, connectivity_pairs,
    state_labels_windows, best_k,
    eng, coherence, avg_entropy,
    memory_score, mem_peaks,
    T, hrf_lag=5,
):
    """Save all 9 plot files into *plots_dir*."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_detailed_timeline(metrics_normalized, viewer_states, moments, T, plots_dir / "timeline.png")
    plot_simple_timeline(metrics_normalized, viewer_states, moments, T, plots_dir / "timeline_simple.png")
    plot_connectivity_states(connectivity_matrices, state_labels_windows, best_k, T, plots_dir / "connectivity.png")
    plot_connectivity_simple(connectivity_matrices, window_centers, state_labels_windows, T, plots_dir / "connectivity_simple.png")
    plot_radar(metrics_normalized, plots_dir / "radar.png")
    plot_coupling(connectivity_pairs, window_centers, T, plots_dir / "coupling.png")
    plot_coupling_simple(connectivity_pairs, window_centers, T, plots_dir / "coupling_simple.png")
    plot_marketing_dashboard(
        metrics_normalized, viewer_states, moments,
        eng, coherence, avg_entropy, T, hrf_lag, plots_dir / "marketing_dashboard.png",
    )
    plot_advanced_analysis(metrics_normalized, memory_score, mem_peaks, T, plots_dir / "advanced_analysis.png")
