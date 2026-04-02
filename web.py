"""
vinD Web — Neural Focus Group analysis frontend.

FastAPI + Jinja2 + Tailwind CSS + HTMX.
No JS frameworks. Pure Python backend with polished, custom UI.

Usage:
    uvicorn web:app --reload
    # or: python web.py
"""

import asyncio
import uuid
import time
import logging
from pathlib import Path

import markdown
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ── App setup ─────────────────────────────────────────────────────────────

app = FastAPI(title="vinD — Neural Focus Group")
BASE = Path(__file__).parent

templates = Jinja2Templates(directory=str(BASE / "templates"))

# Serve results folders as static files
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")

logger = logging.getLogger("vinD.web")

# ── In-memory job store ───────────────────────────────────────────────────

jobs: dict[str, dict] = {}

STAGES = [
    "Sending video to Modal GPU…",
    "Downloading video on server…",
    "Extracting audio & transcribing speech…",
    "Extracting visual features (V-JEPA2)…",
    "Extracting audio features (Wav2Vec-BERT)…",
    "Extracting text features (LLaMA 3.2)…",
    "Running brain prediction model…",
    "Receiving predictions…",
    "Running vinD neural analysis…",
    "Generating plots & report…",
]


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/analyze")
async def start_analysis(request: Request):
    form = await request.form()
    video_url = form.get("video_url", "").strip()

    if not video_url:
        return templates.TemplateResponse(
            request, "partials/error.html",
            context={"message": "Please enter a video URL."},
        )

    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {
        "status": "running",
        "stage": 0,
        "video_url": video_url,
        "started_at": time.time(),
        "error": None,
        "output_dir": None,
    }

    # Fire and forget the background task
    asyncio.create_task(_run_analysis(job_id, video_url))

    return templates.TemplateResponse(
        request, "partials/progress.html",
        context={"job_id": job_id, "stage_text": STAGES[0], "progress": 5},
    )


@app.get("/status/{job_id}", response_class=HTMLResponse)
async def poll_status(request: Request, job_id: str):
    job = jobs.get(job_id)
    if not job:
        return HTMLResponse("<p class='text-red-400'>Job not found.</p>")

    if job["status"] == "complete":
        return templates.TemplateResponse(
            request, "partials/done.html",
            context={"job_id": job_id},
        )

    if job["status"] == "error":
        return templates.TemplateResponse(
            request, "partials/error.html",
            context={"message": job["error"]},
        )

    stage = min(job["stage"], len(STAGES) - 1)
    progress = max(5, int((stage / len(STAGES)) * 95))
    elapsed = int(time.time() - job["started_at"])

    return templates.TemplateResponse(
        request, "partials/progress.html",
        context={
            "job_id": job_id,
            "stage_text": STAGES[stage],
            "progress": progress,
            "elapsed": elapsed,
        },
    )


@app.get("/results/{job_id}", response_class=HTMLResponse)
async def results_page(request: Request, job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "complete":
        return RedirectResponse("/")

    output_dir = Path(job["output_dir"])
    report_path = output_dir / "report.md"

    # Read and render markdown report
    report_md = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    report_html = markdown.markdown(
        report_md,
        extensions=["tables", "fenced_code", "nl2br"],
    )

    # Plot files (the 5 requested)
    plot_names = [
        ("timeline_simple.png", "Viewer Experience Timeline", "Engagement, viewer state, and novelty over time"),
        ("radar.png", "Engagement Radar", "Multi-dimensional engagement profile"),
        ("connectivity_simple.png", "Brain Connectivity States", "Functional connectivity patterns across time"),
        ("coupling_simple.png", "Network Coupling", "Inter-network coupling dynamics"),
        ("advanced_analysis.png", "Advanced Analysis", "Memory encoding, entropy, and state transitions"),
    ]

    plots = []
    for fname, title, desc in plot_names:
        p = output_dir / "plots" / fname
        if p.exists():
            plots.append({
                "url": f"/job-files/{job_id}/plots/{fname}",
                "title": title,
                "description": desc,
            })

    # Parse some headline numbers from the report
    headlines = _extract_headlines(report_md)

    return templates.TemplateResponse(
        request, "results.html",
        context={
            "job_id": job_id,
            "video_url": job["video_url"],
            "plots": plots,
            "report_html": report_html,
            "headlines": headlines,
            "elapsed": int(job.get("elapsed", 0)),
        },
    )


@app.get("/job-files/{job_id}/{path:path}")
async def serve_job_file(job_id: str, path: str):
    """Serve files from a job's output directory."""
    job = jobs.get(job_id)
    if not job or not job.get("output_dir"):
        return HTMLResponse("Not found", status_code=404)

    file_path = Path(job["output_dir"]) / path
    if not file_path.exists() or not file_path.is_file():
        return HTMLResponse("Not found", status_code=404)

    # Ensure the resolved path is within the output directory
    try:
        file_path.resolve().relative_to(Path(job["output_dir"]).resolve())
    except ValueError:
        return HTMLResponse("Forbidden", status_code=403)

    from fastapi.responses import FileResponse
    return FileResponse(file_path)


# ── Startup: register existing results as demo ───────────────────────────

@app.on_event("startup")
async def _register_existing_results():
    """Auto-register any existing results_e2e or jobs/* directories as viewable jobs."""
    e2e = BASE / "results_e2e"
    if e2e.exists() and (e2e / "report.md").exists():
        jobs["demo"] = {
            "status": "complete",
            "video_url": "https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4",
            "started_at": time.time(),
            "elapsed": 0,
            "output_dir": str(e2e),
            "error": None,
            "stage": 10,
        }
        logger.info("Registered demo job from results_e2e/")


# ── Background processing ────────────────────────────────────────────────

async def _run_analysis(job_id: str, video_url: str):
    """Run Modal prediction + vinD analysis in background."""
    import modal

    job = jobs[job_id]
    output_dir = BASE / "jobs" / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        job["stage"] = 1  # downloading
        cls = modal.Cls.from_name("tribe-v2", "TribePredictor")
        predictor = cls()

        job["stage"] = 2  # extracting
        # Run the Modal call in a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: predictor.predict.remote(video_url)
        )

        job["stage"] = 7  # receiving preds
        preds = np.frombuffer(
            result["preds_bytes"], dtype=np.float32
        ).reshape(result["shape"])

        # Save preds
        np.savez_compressed(str(output_dir / "tribe_preds.npz"), preds=preds)
        job["stage"] = 8  # running vinD

        # Run vinD analysis (CPU-bound, run in executor)
        from vinD import run_analysis
        await loop.run_in_executor(
            None, lambda: run_analysis(preds, output_dir=str(output_dir))
        )

        job["stage"] = 9  # generating plots
        job["status"] = "complete"
        job["output_dir"] = str(output_dir)
        job["elapsed"] = int(time.time() - job["started_at"])

    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        job["status"] = "error"
        job["error"] = str(e)


# ── Helpers ───────────────────────────────────────────────────────────────

def _extract_headlines(report_md: str) -> dict:
    """Pull key numbers from the markdown report for the hero cards."""
    import re

    h = {
        "total_engagement": "—",
        "emotional": "—",
        "cognitive": "—",
        "sensory": "—",
        "social": "—",
        "retention": "—",
        "duration": "—",
        "hook": "—",
        "ending": "—",
        "entropy": "—",
        "brain_states": "—",
        "moments": "—",
    }

    for line in report_md.split("\n"):
        line_stripped = line.strip()
        if "Total Engagement" in line_stripped:
            m = re.search(r"(\d+\.?\d*)", line_stripped.split("Total Engagement")[-1])
            if m: h["total_engagement"] = m.group(1)
        elif "Emotional Engagement" in line_stripped:
            m = re.search(r"(\d+\.?\d*)", line_stripped.split("Emotional Engagement")[-1])
            if m: h["emotional"] = m.group(1)
        elif "Cognitive Engagement" in line_stripped:
            m = re.search(r"(\d+\.?\d*)", line_stripped.split("Cognitive Engagement")[-1])
            if m: h["cognitive"] = m.group(1)
        elif "Sensory Engagement" in line_stripped:
            m = re.search(r"(\d+\.?\d*)", line_stripped.split("Sensory Engagement")[-1])
            if m: h["sensory"] = m.group(1)
        elif "Social Connection" in line_stripped:
            m = re.search(r"(\d+\.?\d*)", line_stripped.split("Social Connection")[-1])
            if m: h["social"] = m.group(1)
        elif "Predicted Retention" in line_stripped:
            m = re.search(r"(\d+\.?\d*)", line_stripped.split("Predicted Retention")[-1])
            if m: h["retention"] = m.group(1)
        elif "Video duration" in line_stripped:
            m = re.search(r"(\d+)", line_stripped.split("Video duration")[-1])
            if m: h["duration"] = m.group(1) + "s"
        elif "Hook Strength" in line_stripped:
            m = re.search(r"(\d+\.?\d*)/100", line_stripped.split("Hook Strength")[-1])
            if m: h["hook"] = m.group(1)
        elif "Ending Strength" in line_stripped:
            m = re.search(r"(\d+\.?\d*)/100", line_stripped.split("Ending Strength")[-1])
            if m: h["ending"] = m.group(1)
        elif "Brain Dynamism" in line_stripped:
            m = re.search(r"(\d+\.?\d*)", line_stripped.split("Brain Dynamism")[-1])
            if m: h["entropy"] = m.group(1)
        elif "distinct brain states" in line_stripped:
            m = re.search(r"(\d+)", line_stripped)
            if m: h["brain_states"] = m.group(1)
        elif "moments detected" in line_stripped:
            m = re.search(r"(\d+)", line_stripped)
            if m: h["moments"] = m.group(1)

    return h


# ── Dev server ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web:app", host="127.0.0.1", port=8000, reload=True)
