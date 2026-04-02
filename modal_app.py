"""
TRIBE v2 on Modal — GPU inference for brain activity prediction from video.

Supports YouTube, Vimeo, Twitter/X, TikTok, direct URLs, and 1000+ sites via yt-dlp.

Usage:
    # One-off run (ephemeral, no deployment needed):
    modal run modal_app.py --video-url "https://www.youtube.com/watch?v=..."

    # Or deploy first, then call repeatedly from run_tribe.py:
    modal deploy modal_app.py
    python run_tribe.py "https://www.youtube.com/watch?v=..."

Setup:
    1. modal secret create huggingface-secret HF_TOKEN=hf_YOUR_TOKEN_HERE
    2. pip install modal  (or uv add modal)
"""

import modal
import subprocess
import tempfile
from pathlib import Path

app = modal.App("tribe-v2")

# Persistent volume for model weights + HuggingFace cache
cache_vol = modal.Volume.from_name("tribe-v2-cache", create_if_missing=True)
CACHE_PATH = "/cache"

# ---------------------------------------------------------------------------
# Container image — layered for efficient caching
# ---------------------------------------------------------------------------
tribe_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "libsndfile1")
    # Heavy ML deps first (cached across rebuilds)
    .pip_install(
        "torch>=2.5.1,<2.7",
        "torchvision>=0.20,<0.22",
        "torchaudio",
    )
    # Video downloader
    .pip_install("yt-dlp")
    # TRIBE v2 + all its deps (transformers, moviepy, spacy, etc.)
    .pip_install(
        "tribev2 @ git+https://github.com/facebookresearch/tribev2.git",
    )
    .env({
        "HF_HOME": f"{CACHE_PATH}/huggingface",
        "TORCH_HOME": f"{CACHE_PATH}/torch",
    })
)


# ---------------------------------------------------------------------------
# Video download helpers
# ---------------------------------------------------------------------------

def _download_video(url: str, tmpdir: str) -> Path:
    """Download video from URL.

    Tries yt-dlp first (handles YouTube, Vimeo, Twitter/X, TikTok, direct
    links, and 1000+ other sites). Falls back to a plain HTTP download for
    simple direct-file URLs.
    """
    import glob

    output_template = str(Path(tmpdir) / "video.%(ext)s")
    output_mp4 = Path(tmpdir) / "video.mp4"

    # --- yt-dlp attempt ---------------------------------------------------
    try:
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--no-part",
            "-S", "res:1080,ext:mp4:m4a",
            "--merge-output-format", "mp4",
            "--recode-video", "mp4",
            "-o", output_template,
            url,
        ]
        print(f"[download] yt-dlp {url}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        print(f"[download] yt-dlp stdout: {result.stdout[-500:]}")
        if result.returncode != 0:
            print(f"[download] yt-dlp stderr: {result.stderr[-500:]}")

        # Find the actual output file (yt-dlp may rename)
        candidates = glob.glob(str(Path(tmpdir) / "video.*"))
        video_files = [f for f in candidates if not f.endswith(".part")]
        if video_files:
            chosen = video_files[0]
            size_mb = Path(chosen).stat().st_size / 1e6
            print(f"[download] OK via yt-dlp — {chosen} ({size_mb:.1f} MB)")
            return Path(chosen)

        print(f"[download] yt-dlp produced no output file, falling back")
    except subprocess.TimeoutExpired:
        print("[download] yt-dlp timed out, falling back to direct download")
    except FileNotFoundError:
        print("[download] yt-dlp not found, falling back to direct download")

    # --- Direct HTTP fallback ---------------------------------------------
    import requests
    from urllib.parse import urlparse

    print(f"[download] direct HTTP {url}")
    resp = requests.get(url, stream=True, timeout=120, allow_redirects=True)
    resp.raise_for_status()

    ext = Path(urlparse(url).path).suffix or ".mp4"
    dl_path = Path(tmpdir) / f"video{ext}"

    total = 0
    with open(dl_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=256 * 1024):
            f.write(chunk)
            total += len(chunk)
    print(f"[download] OK — {total / 1e6:.1f} MB")
    return dl_path


def _serialize_segments(segments) -> list:
    """Best-effort conversion of neuralset segment objects to plain dicts."""
    result = []
    for seg in segments:
        try:
            d = {}
            if hasattr(seg, "__dict__"):
                for k, v in seg.__dict__.items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        d[k] = v
            result.append(d)
        except Exception:
            result.append({})
    return result


# ---------------------------------------------------------------------------
# Modal class — loads model once, serves predictions
# ---------------------------------------------------------------------------

@app.cls(
    image=tribe_image,
    gpu="A100-80GB",
    timeout=900,  # 15 min
    volumes={CACHE_PATH: cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class TribePredictor:
    @modal.enter()
    def load_model(self):
        from tribev2.demo_utils import TribeModel

        print("[model] Loading TRIBE v2 …")
        self.model = TribeModel.from_pretrained(
            "facebook/tribev2",
            cache_folder=f"{CACHE_PATH}/tribe_model",
        )
        cache_vol.commit()  # persist downloaded weights
        print("[model] Ready.")

    @modal.method()
    def predict(self, video_url: str) -> dict:
        """Download video and predict brain activity with TRIBE v2.

        Returns dict with keys:
            preds_bytes : bytes   — raw float32 array
            dtype       : str
            shape       : list[int]
            segments    : list[dict]
        """
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = _download_video(video_url, tmpdir)
            print(f"[predict] Building events dataframe …")
            df = self.model.get_events_dataframe(video_path=str(video_path))
            print(f"[predict] Events: {len(df)} rows.  Running model …")
            preds, segments = self.model.predict(events=df)
            print(f"[predict] Done — preds {preds.shape} ({preds.dtype})")

            return {
                "preds_bytes": preds.astype(np.float32).tobytes(),
                "dtype": "float32",
                "shape": list(preds.shape),
                "segments": _serialize_segments(segments),
            }


# ---------------------------------------------------------------------------
# Local entrypoint — `modal run modal_app.py --video-url "…"`
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(video_url: str, output_dir: str = "results"):
    """Run TRIBE v2 on a video and analyze locally with vinD."""
    import numpy as np

    print(f"→ Sending to Modal for TRIBE v2 prediction …")
    print(f"  Video: {video_url}")
    predictor = TribePredictor()
    result = predictor.predict.remote(video_url)

    # Reconstruct numpy array from bytes
    preds = np.frombuffer(result["preds_bytes"], dtype=np.float32).reshape(result["shape"])
    print(f"← Received predictions: {preds.shape}")

    # Save raw predictions
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz_path = out / "tribe_preds.npz"
    np.savez_compressed(str(npz_path), preds=preds)
    print(f"   Saved predictions → {npz_path}")

    # Run vinD analysis
    from vinD import run_analysis

    run_analysis(preds, output_dir=output_dir)
    print(f"✓ Done — results in {output_dir}/")
