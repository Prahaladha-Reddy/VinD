import numpy as np
import pandas as pd
from scipy.stats import zscore


STATE_COLORS = {
    "visually_absorbed": "#60a5fa",
    "emotionally_gripped": "#f87171",
    "thinking_hard": "#a78bfa",
    "socially_connecting": "#f472b6",
    "personally_relating": "#34d399",
    "mind_wandering": "#525252",
    "confused": "#fbbf24",
    "checked_out": "#404040",
}

FRIENDLY_NAMES = {
    "visually_absorbed": "Eyes locked",
    "emotionally_gripped": "Feeling it",
    "thinking_hard": "Thinking",
    "socially_connecting": "Connecting",
    "personally_relating": "Relates to me",
    "mind_wandering": "Zoning out",
    "confused": "Confused",
    "checked_out": "Checked out",
}


def compute_metrics(network_ts, key_regions):
    """Compute 15+ per-second engagement/arousal/etc metrics.

    Returns
    -------
    metrics : pd.DataFrame
        Raw z-scored metrics.
    metrics_normalized : pd.DataFrame
        All columns scaled 0-100.
    """
    T = len(network_ts["visual"])

    z_nets = {k: zscore(v) if v.std() > 0 else v for k, v in network_ts.items()}
    z_regions = {k: zscore(v) if v.std() > 0 else v for k, v in key_regions.items()}

    metrics = pd.DataFrame(index=range(T))

    # Four types of engagement
    metrics["sensory_engagement"] = (z_nets["visual"] + z_nets["dorsal_attention"]) / 2
    metrics["emotional_engagement"] = (z_nets["limbic"] + z_nets["ventral_attention"]) / 2
    metrics["cognitive_engagement"] = z_nets["frontoparietal"]
    metrics["social_engagement"] = z_nets["ventral_attention"]

    # Mind-wandering: DMN active while all other engagement types are below median
    all_engagement = np.column_stack([
        metrics["sensory_engagement"],
        metrics["emotional_engagement"],
        metrics["cognitive_engagement"],
    ])
    max_engagement = all_engagement.max(axis=1)
    dmn_z = z_nets["default_mode"]
    metrics["mind_wandering"] = np.where(
        (max_engagement < 0) & (dmn_z > 0), dmn_z, 0
    )

    # Narrative engagement: DMN co-activating with task-positive networks
    metrics["narrative_engagement"] = np.where(
        max_engagement > 0, np.clip(dmn_z, 0, None), 0
    )

    # Total engagement: max across all types, minus mind-wandering
    metrics["total_engagement"] = (
        np.maximum.reduce([
            metrics["sensory_engagement"],
            metrics["emotional_engagement"],
            metrics["cognitive_engagement"],
            metrics["narrative_engagement"],
        ])
        - metrics["mind_wandering"]
    )

    # Specific signals
    metrics["novelty"] = np.abs(np.gradient(z_nets["visual"]))
    metrics["face_response"] = z_regions["face_processing"]
    metrics["language_response"] = (
        z_regions["language_broca"] + z_regions["language_temporal"]
    ) / 2
    metrics["memory_encoding"] = z_regions["memory_encoding"]
    metrics["reward_signal"] = z_regions["reward_anticipation"]
    metrics["attention_control"] = z_regions["attention_control"]

    # Arousal: absolute magnitude of limbic response
    metrics["arousal"] = np.abs(z_nets["limbic"])

    # Valence proxy
    metrics["valence_proxy"] = z_regions["reward_anticipation"] - np.clip(
        -z_nets["limbic"], 0, None
    )

    # Normalize all to 0-100
    metrics_normalized = metrics.copy()
    for col in metrics_normalized.columns:
        vals = metrics_normalized[col].values
        vmin, vmax = vals.min(), vals.max()
        if vmax != vmin:
            metrics_normalized[col] = (
                (vals - vmin) / (vmax - vmin) * 100
            ).round(1)
        else:
            metrics_normalized[col] = 50.0

    return metrics, metrics_normalized


def _classify_single_state(row):
    """Classify a single second into a dominant viewer state."""
    states = {
        "visually_absorbed": row["sensory_engagement"],
        "emotionally_gripped": row["emotional_engagement"],
        "thinking_hard": row["cognitive_engagement"],
        "socially_connecting": row["social_engagement"],
        "personally_relating": row["narrative_engagement"],
    }

    if row["mind_wandering"] > 0.5:
        return "mind_wandering"

    if (
        row["cognitive_engagement"] > 0.5
        and row["sensory_engagement"] < 0
        and row["emotional_engagement"] < 0
    ):
        return "confused"

    dominant = max(states, key=states.get)
    if states[dominant] < -0.5:
        return "checked_out"

    return dominant


def classify_viewer_states(metrics_raw):
    """Classify each second into one of 8 viewer states.

    Returns
    -------
    viewer_states : pd.Series (T,)
    """
    return metrics_raw.apply(_classify_single_state, axis=1)
