# NZAP Engine injects a `params` dict before this script runs.
# Vocal remover and music stem splitter with Demucs (HT Demucs), packaged as
# an NZAP app (see APPS.md).
#
# The first run installs Demucs and loads the chosen model; models stay in the
# kernel, so later runs on the same runtime go straight to separation. Audio
# is decoded and encoded with ffmpeg and soundfile, so torchaudio's changing
# I/O backends never get in the way. Stems are written to
# /content/nzap/outputs/stem-splitter/.

import math
import os
import re
import subprocess
import sys
import time

from IPython.display import display

APP = "stem-splitter"
OUT_DIR = f"/content/nzap/outputs/{APP}"
# demucs 4.1.0 loads its weights as safetensors from the Hugging Face Hub and
# needs no torchaudio; it uses the runtime's preinstalled torch.
PACKAGES = ("demucs==4.1.0",)
# Official Demucs weights (MIT), pinned to a commit.
MODELS = {
    "htdemucs": ("adefossez/HTDemucs", "cbc8a9b1a87023b7fd74e7b3412e6321c0eab003"),
    "htdemucs_ft": ("adefossez/HTDemucs-ft", "d74ac89c3a1e874fc78f152555cf4d8533f06cd4"),
}

# Colab's huggingface_hub otherwise waits on the notebook-UI secret store.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")


def nzap(event, console, **fields):
    """One NZAP app event; `console` is the `text/plain` line the console shows
    (not named `text`, which a text output uses as a field)."""
    payload = {"v": 1, "app": APP, "event": event, **fields}
    display({"application/vnd.nzap.app+json": payload, "text/plain": console}, raw=True)


def stage(stage_id, label, progress=None):
    if progress is None:
        nzap("stage", f"[nzap] {label}…", id=stage_id, label=label)
    else:
        nzap("stage", f"[nzap] {label}… {progress:.0%}", id=stage_id, label=label, progress=round(progress, 3))


def pip(*packages):
    command = [sys.executable, "-m", "pip", "install", "-q", *packages]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"pip install {' '.join(packages)} failed:\n{result.stderr[-2000:]}")


def decode(path, rate):
    """Any audio or video file -> stereo float32 at the model's sample rate, via ffmpeg."""
    import numpy as np

    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "2", "-ar", str(rate), "-f", "f32le", "-"]
    result = subprocess.run(command, capture_output=True)
    if result.returncode:
        reason = result.stderr.decode(errors="replace").strip().splitlines()[-1:] or ["unknown error"]
        raise RuntimeError(f"ffmpeg could not read {os.path.basename(path)}: {reason[0]}")
    audio = np.frombuffer(result.stdout, dtype=np.float32).reshape(-1, 2).T.copy()
    if audio.shape[1] < rate:
        raise RuntimeError(f"{os.path.basename(path)} has less than a second of audio to split.")
    return audio


def encode(path, audio, rate, fmt):
    """(2, samples) float32 -> a 16-bit WAV (soundfile) or a 320 kbps MP3 (ffmpeg)."""
    import numpy as np
    import soundfile as sf

    audio = np.clip(audio, -1.0, 1.0).T.astype(np.float32)
    if fmt == "wav":
        sf.write(path, audio, rate, subtype="PCM_16")
        return
    command = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "f32le", "-ar", str(rate), "-ac", "2", "-i", "-",
               "-c:a", "libmp3lame", "-b:a", "320k", path]
    result = subprocess.run(command, input=audio.tobytes(), capture_output=True)
    if result.returncode:
        raise RuntimeError(f"ffmpeg could not write the MP3: {result.stderr.decode(errors='replace')[-300:]}")


def load_bag(name):
    """The model's bag (one or four networks) from its pinned Hub commit."""
    import yaml
    from demucs.apply import BagOfModels
    from demucs.hf import load_safetensors_model
    from huggingface_hub import hf_hub_download

    repo, revision = MODELS[name]
    with open(hf_hub_download(repo, f"{name}.yaml", revision=revision)) as file:
        spec = yaml.safe_load(file)
    nets = [load_safetensors_model(hf_hub_download(repo, f"{sig}.safetensors", revision=revision)) for sig in spec["models"]]
    return BagOfModels(nets, spec.get("weights"), spec.get("segment")).eval()


started = time.time()
state = globals().setdefault("_nzap_apps", {}).setdefault(APP, {})
models = state.setdefault("models", {})
name = params.get("model", "htdemucs")
if name not in MODELS:
    raise RuntimeError(f"Unknown model {name!r}; pick htdemucs or htdemucs_ft.")
warm = name in models

source = params.get("audio", "")
if not source or not os.path.isfile(source):
    raise RuntimeError("Upload a song (or a video with music) to split.")

if not warm:
    if not state.get("installed"):
        stage("install", "Installing Demucs")
        pip(*PACKAGES)
        state["installed"] = True
    stage("download", f"Downloading and loading {name}")
    import torch

    state["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    models[name] = load_bag(name)

import numpy as np
import torch
from demucs.apply import apply_model

bag, device = models[name], state["device"]
setup_seconds = time.time() - started
nzap("ready", f"[nzap] {name} ready on {device}.", warm=warm, setupSeconds=round(setup_seconds, 2), device=device)

run_started = time.time()
stage("run", "Reading the audio", 0)
mix = decode(source, bag.samplerate)
duration = mix.shape[1] / bag.samplerate
limit = float(params.get("max_minutes", 10))
if limit > 0 and duration > limit * 60:
    raise RuntimeError(
        f"The track is {duration / 60:.1f} minutes long, over the {limit:g}-minute limit; trim it or raise Max minutes."
    )

sources = list(bag.sources)  # drums, bass, other, vocals
vocals_at = sources.index("vocals")
two_stems = params.get("mode", "vocals") == "vocals"
# htdemucs_ft is four specialists, one per stem. For vocals + instrumental only
# the vocal specialist is needed, so it runs at the speed of plain htdemucs.
nets, weights = list(bag.models), bag.weights
specialist = [i for i, w in enumerate(weights) if w[vocals_at] and not any(v for k, v in enumerate(w) if k != vocals_at)]
if two_stems and len(nets) > 1 and len(specialist) == 1:
    model = nets[specialist[0]]
    runs = 1
else:
    model = bag
    runs = len(nets)

# Progress: Demucs reports each ~7.8 s window it finishes (25% overlap).
segment = float(nets[0].segment)
windows = runs * math.ceil((mix.shape[1] + bag.samplerate / 2) / int(0.75 * segment * bag.samplerate))
tally = {"done": 0, "reported": 0.0}


def on_window(info):
    if info.get("state") == "end":
        tally["done"] += 1
        if time.time() - tally["reported"] > 2.0:
            tally["reported"] = time.time()
            stage("run", f"Separating ({name}, {device})", min(tally["done"] / windows, 0.99))


stage("run", f"Separating ({name}, {device})", 0)
# Demucs works on audio normalised to zero mean and unit variance.
wav = torch.from_numpy(mix)
ref = wav.mean(0)
mean, std = ref.mean(), ref.std() + 1e-8
with torch.inference_mode():
    estimates = apply_model(model, ((wav - mean) / std)[None], device=device, shifts=1, split=True, overlap=0.25, callback=on_window)
estimates = (estimates[0] * std + mean).cpu().numpy()  # (sources, 2, samples)

vocals = estimates[vocals_at]
if model is bag:
    stems = {source_name: estimates[i] for i, source_name in enumerate(sources)}
    # Same as `demucs --two-stems vocals`: everything else, summed.
    stems["instrumental"] = sum(estimates[i] for i in range(len(sources)) if i != vocals_at)
else:
    stems = {"vocals": vocals, "instrumental": mix - vocals}
order = ["vocals", "instrumental"] if two_stems else ["vocals", "drums", "bass", "other", "instrumental"]

fmt = params.get("format", "mp3")
os.makedirs(OUT_DIR, exist_ok=True)
title = re.sub(r"[^A-Za-z0-9_-]+", "-", os.path.splitext(os.path.basename(source))[0]).strip("-")[:60] or "track"
stamp = time.strftime("%Y%m%d-%H%M%S")
stage("run", "Saving the stems", 0.99)
paths = {}
for stem in order:
    paths[stem] = f"{OUT_DIR}/{stamp}-{title}-{stem}.{fmt}"
    encode(paths[stem], stems[stem], bag.samplerate, fmt)
run_seconds = time.time() - run_started

for stem in order:
    level = 20 * np.log10(np.sqrt(np.mean(stems[stem] ** 2)) + 1e-9)
    nzap(
        "output",
        f"[nzap] Saved {paths[stem]} ({level:.1f} dBFS RMS)",
        id=stem,
        kind="audio",
        path=paths[stem],
        mime="audio/mpeg" if fmt == "mp3" else "audio/wav",
        meta={"duration": round(duration, 2), "sampleRate": bag.samplerate, "rmsDb": round(float(level), 1)},
    )
nzap(
    "done",
    f"[nzap] Done in {time.time() - started:.1f}s ({duration:.0f}s of audio, {run_seconds / duration:.2f}s per second of audio on {device}).",
    seconds={"setup": round(setup_seconds, 2), "run": round(run_seconds, 2)},
    warm=warm,
)
