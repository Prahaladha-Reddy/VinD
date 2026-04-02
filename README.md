# vinD

Marketing teams spend `$50K+` promoting a video that `60%` of viewers scroll past in the first `3 seconds`.

The editor liked it.  
The internal team liked it.  
The focus group of 8 people liked it.

But none of them are the average viewer, and most teams cannot afford to run a 500-person study for every cut.

`vinD` is the idea I built in response to that gap.

Last week, Meta FAIR released `TRIBE v2`: a model trained on `1,000+` hours of real brain scans from `720` people watching videos, listening to podcasts, and reading text. You give it a piece of content, and it predicts the corresponding fMRI brain response. Not opinions. Not generated reviews. Predicted neural activity, at `20,000+` brain data points per second-level timeline.

Raw brain signals are not useful to a marketing team, so `vinD` adds an interpretation layer on top.

## What vinD does

- Maps `20K+` predicted brain points into `400` known brain regions.
- Tracks `7` major brain networks across time:
  visual, emotional, attention, social, cognitive, default mode, and limbic.
- Measures which networks are coupling in real time:
  are viewers seeing and feeling, or just passively watching?
- Classifies every second into a viewer state such as:
  `eyes locked`, `feeling it`, `personally relating`, `zoning out`, or `confused`.
- Detects likely memorable moments versus forgettable ones.
- Produces a second-by-second viewer experience report with plots, moments, and summary metrics.

The result is a simulated audience-response layer for video: engagement curves, emotional peaks, attention drops, memory windows, and possible CTA timing, generated without a traditional focus group.

Think of it as a simulator for `720 average brains` reacting to your content.

## Early result that motivated this repo

On an early marketing-video test:

- Hook strength: `0/100`
- Emotional engagement: `61/100`
- The model found an emotional peak at `0:29` that was independently verified
- Predicted `47%` of viewing time spent in a `personally relating` state

The signal was clear: the opening was dead, even though nobody in review had flagged it.

That is the core thesis behind `vinD`:
editors evaluate videos based on craft, but viewer brains evaluate videos based on attention, emotion, and personal relevance. Those are not the same thing.

It needs validation against real retention and performance data. But the direction is promising enough to make the workflow concrete.

## Core outputs

Each run can generate:

- `report.md`
- `plots/timeline_simple.png`
- `plots/radar.png`
- `plots/connectivity_simple.png`
- `plots/coupling_simple.png`
- `plots/advanced_analysis.png`

These outputs are meant to answer practical questions such as:

- Does the hook actually hold attention?
- Where does attention collapse?
- Which seconds create emotional engagement?
- Where are viewers likely to remember the content?
- Is the viewer processing, relating, or drifting away?

## Install

Requirements:

- Python `3.12+`
- `uv`
- TRIBE predictions on `fsaverage5` cortical vertices (`20,484` columns minimum)

Install dependencies:

```bash
uv sync
```

Run tests:

```bash
uv run --with pytest pytest
```

## Usage

Analyze an existing predictions file:

```bash
uv run vind-analyze path/to/tribe_preds.npz --output-dir results/demo
```

The `.npz` file must contain a `preds` array.

Run remote TRIBE inference on Modal, then analyze locally:

```bash
uv run vind-tribe "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --output-dir results/demo
```

If your deployed Modal app uses a different name:

```bash
uv run vind-tribe "<video-url>" --modal-app-name my-app --modal-class-name MyPredictor
```

Run the web app:

```bash
uv run uvicorn web:app --reload
```

## Modal setup

Create the Hugging Face secret:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_your_token
```

Deploy the inference worker:

```bash
modal deploy modal_app.py
```

Then use `vind-tribe` or the web app to call it.

## Current limitations

- This is not yet validated against large-scale retention or conversion data.
- The web app keeps job state in memory, so it is suitable for lightweight single-process use, not production queueing.
- Atlas data is downloaded on first run into the local Nilearn cache.
- The interpretation layer is the product idea here; the underlying neural prediction model is `TRIBE v2`.

## Why this matters

Most video review workflows are optimized around taste and craft. `vinD` is an attempt to optimize around predicted audience cognition instead.

If that works, the value is straightforward:

- catch weak hooks before media spend
- find emotional peaks before launch
- compare edits on attention and memorability, not just preference
- reduce reliance on tiny, noisy review groups

## Reference

`TRIBE v2` is based on:

- d'Ascoli et al., `2026`, _A foundation model of vision, audition, and language for in-silico neuroscience_ (Meta FAIR)
