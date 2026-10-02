# NZAP Engine injects a `params` dict before this script runs.
# Classifies the sentiment of one or more lines of text, as an NZAP app
# (see APPS.md): the scores come back as a table.

import os
import subprocess
import sys
import time

from IPython.display import display

APP = "hf-sentiment"
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")


def nzap(event, text, **fields):
    """One NZAP app event; `text/plain` is what the console shows."""
    payload = {"v": 1, "app": APP, "event": event, **fields}
    display({"application/vnd.nzap.app+json": payload, "text/plain": text}, raw=True)


started = time.time()
state = globals().setdefault("_nzap_apps", {}).setdefault(APP, {})
warm = "classifier" in state
if not warm:
    nzap("stage", "[nzap] Loading DistilBERT…", id="load", label="Loading DistilBERT")
    try:
        import transformers  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers"])

    import torch
    from transformers import pipeline

    state["classifier"] = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if torch.cuda.is_available() else -1,
    )
setup_seconds = time.time() - started
nzap("ready", "[nzap] Model ready.", warm=warm, setupSeconds=round(setup_seconds, 2))

nzap("stage", "[nzap] Scoring…", id="run", label="Scoring")
run_started = time.time()
lines = [line.strip() for line in params["text"].splitlines() if line.strip()]
results = state["classifier"](lines)
rows = []
for line, result in zip(lines, results):
    print(f"{result['label']:>8} {result['score']:.3f}  {line}")
    rows.append([line, result["label"].title(), round(result["score"], 4)])

nzap(
    "output",
    f"[nzap] Scored {len(rows)} lines.",
    id="scores",
    kind="table",
    columns=["Text", "Sentiment", "Confidence"],
    rows=rows,
)
nzap(
    "done",
    f"[nzap] Done in {time.time() - started:.1f}s.",
    seconds={"setup": round(setup_seconds, 2), "run": round(time.time() - run_started, 2)},
    warm=warm,
)
