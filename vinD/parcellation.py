import numpy as np
import nibabel as nib
import os
import urllib.request
from pathlib import Path


NETWORK_LABELS = {
    "Vis": "visual",
    "SomMot": "somatomotor",
    "DorsAttn": "dorsal_attention",
    "SalVentAttn": "ventral_attention",
    "Limbic": "limbic",
    "Cont": "frontoparietal",
    "Default": "default_mode",
}


def load_schaefer_atlas(n_parcels=400, n_networks=7):
    """Download and load Schaefer surface parcellation for fsaverage5.

    Returns
    -------
    vertex_labels : np.ndarray (20484,)
        Per-vertex parcel ID (1-indexed, 0 = background).
    parcel_names : list[str]
        Human-readable name for each parcel.
    """
    cache_dir = Path(os.path.expanduser("~")) / "nilearn_data" / "schaefer_surf"
    cache_dir.mkdir(parents=True, exist_ok=True)

    base_url = (
        "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
        "stable_projects/brain_parcellation/"
        "Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/"
        "fsaverage5/label/"
    )

    vertex_labels = []
    parcel_names = []
    parcel_counter = 0

    for hemi in ["lh", "rh"]:
        fname = f"{hemi}.Schaefer2018_{n_parcels}Parcels_{n_networks}Networks_order.annot"
        local_path = cache_dir / fname
        if not local_path.exists():
            print(f"  Downloading {fname}...")
            urllib.request.urlretrieve(base_url + fname, local_path)

        labels, ctab, names = nib.freesurfer.read_annot(str(local_path))
        names = [n.decode() if isinstance(n, bytes) else n for n in names]

        remapped = np.zeros(len(labels), dtype=int)
        for local_idx, name in enumerate(names):
            if "Background" in name or "Medial_Wall" in name or name == "???":
                continue
            mask = labels == local_idx
            if mask.sum() > 0:
                parcel_counter += 1
                remapped[mask] = parcel_counter
                parcel_names.append(name)

        vertex_labels.append(remapped)

    vertex_labels = np.concatenate(vertex_labels)
    return vertex_labels, parcel_names


def parcellate_predictions(preds, vertex_labels, parcel_names):
    """Average vertex-level predictions within each parcel.

    Returns
    -------
    parcel_ts : np.ndarray (T, n_parcels)
    """
    T = preds.shape[0]
    n_parcels = len(parcel_names)
    parcel_ts = np.zeros((T, n_parcels))

    for i in range(n_parcels):
        pid = i + 1
        mask = vertex_labels == pid
        if mask.sum() > 0:
            parcel_ts[:, i] = preds[:, mask].mean(axis=1)

    return parcel_ts


def compute_network_timeseries(parcel_ts, parcel_names):
    """Average parcel timeseries within each of the 7 Yeo networks.

    Returns
    -------
    network_ts : dict[str, np.ndarray (T,)]
    """
    T = parcel_ts.shape[0]

    parcel_network = []
    for name in parcel_names:
        assigned = "unknown"
        for key, net in NETWORK_LABELS.items():
            if key in name:
                assigned = net
                break
        parcel_network.append(assigned)

    network_ts = {}
    for net_name in NETWORK_LABELS.values():
        cols = [i for i, n in enumerate(parcel_network) if n == net_name]
        if cols:
            network_ts[net_name] = parcel_ts[:, cols].mean(axis=1)
        else:
            network_ts[net_name] = np.zeros(T)

    return network_ts


def extract_key_regions(parcel_ts, parcel_names):
    """Extract timeseries for 10 key brain regions by name matching.

    Returns
    -------
    key_regions : dict[str, np.ndarray (T,)]
    """
    T = parcel_ts.shape[0]

    def _find_parcels(keyword, hemisphere=None):
        results = []
        for i, name in enumerate(parcel_names):
            if keyword in name:
                if hemisphere is None:
                    results.append(i)
                elif hemisphere == "L" and "_LH_" in name:
                    results.append(i)
                elif hemisphere == "R" and "_RH_" in name:
                    results.append(i)
        return results

    def _region_ts(keyword, hemisphere=None):
        idx = _find_parcels(keyword, hemisphere)
        if idx:
            return parcel_ts[:, idx].mean(axis=1)
        return np.zeros(T)

    key_regions = {
        "early_visual": _region_ts("Vis_"),
        "motion_area": _region_ts("DorsAttn_FEF"),
        "face_processing": _region_ts("SalVentAttn_TempOcc"),
        "social_cognition": _region_ts("Default_PFC"),
        "language_broca": _region_ts("Cont_PFCl", "L"),
        "language_temporal": _region_ts("Default_Temp", "L"),
        "executive_control": _region_ts("Cont_PFCl"),
        "attention_control": _region_ts("DorsAttn_"),
        "memory_encoding": _region_ts("Default_PCC"),
        "reward_anticipation": _region_ts("Limbic_OFC"),
    }

    return key_regions
