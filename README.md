# vinD

`vinD` turns TRIBE brain-activity predictions into a publishable analysis package: per-second engagement metrics, dynamic connectivity states, critical moment detection, report generation, and a lightweight FastAPI viewer.

## What it does

- Analyzes TRIBE prediction arrays shaped `(time, vertices)`.
- Parcellates fsaverage5 cortical predictions into Schaefer 400 / Yeo 7 network summaries.
- Computes engagement, arousal, novelty, memorability, connectivity, and state-transition signals.
- Generates a Markdown report plus plot assets.
- Optionally calls a deployed Modal TRIBE worker from a local CLI.
- Includes a small web app for submitting jobs and browsing results.

## Requirements

- Python `3.12+`
- `uv` recommended for dependency management
- TRIBE predictions on `fsaverage5` cortical vertices (`20,484` columns minimum)

## Install

```bash
uv sync
```

For tests:

```bash
uv run --with pytest pytest
```

## Usage

Run analysis on an existing predictions file:

```bash
uv run vind-analyze path/to/tribe_preds.npz --output-dir results/demo
```

The `.npz` file must contain a `preds` array.

Run TRIBE remotely on Modal, then analyze locally:

```bash
uv run vind-tribe "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --output-dir results/demo
```

If your deployed Modal names differ, override them:

```bash
uv run vind-tribe "<video-url>" --modal-app-name my-app --modal-class-name MyPredictor
```

Start the web app locally:

```bash
uv run uvicorn web:app --reload
```

## Output

Each run writes:

- `report.md`
- `plots/`
- `tribe_preds.npz` when using the Modal client

Typical plot outputs include:

- `timeline_simple.png`
- `radar.png`
- `connectivity_simple.png`
- `coupling_simple.png`
- `advanced_analysis.png`

## Modal setup

Create the secret once:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_your_token
```

Deploy the inference app:

```bash
modal deploy modal_app.py
```

Then call it from `vind-tribe`.

## Publish checklist

Before pushing to GitHub:

1. Run `uv sync`.
2. Run `uv run --with pytest pytest`.
3. Smoke-test one CLI path:
   `uv run vind-analyze <predictions.npz> --output-dir results/smoke`
4. If you plan to demo remote inference, verify `modal deploy modal_app.py` succeeds in your account.
5. Review `git status` so you do not accidentally commit large result files or local caches.

## Notes

- The analysis pipeline now validates input shape early and handles short time series more defensibly.
- The web app keeps job state in memory, so it is suitable for a lightweight single-process deployment, not a multi-instance queueing setup.
- Atlas files are downloaded into the local Nilearn cache on first run.
