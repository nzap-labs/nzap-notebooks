# NZAP Engine injects a `params` dict before this script runs.
# Kokoro-82M text to speech, packaged as an NZAP app (see APPS.md).
#
# The first run on a runtime installs Kokoro and loads the model; the model
# stays in the kernel, so every later run on the same runtime only
# synthesises. The audio is written to /content/nzap/outputs/kokoro-tts/.

import os
import subprocess
import sys
import time

from IPython.display import display

APP = "kokoro-tts"
OUT_DIR = f"/content/nzap/outputs/{APP}"
SAMPLE_RATE = 24_000
# misaki[en]'s runtime dependencies, minus spacy-curated-transformers.
BASE_DEPS = ("loguru", "addict", "regex", "num2words", "spacy", "phonemizer-fork", "espeakng-loader", "soundfile")
# Japanese and Mandarin need extra phonemisers, installed the first time.
EXTRAS = {
    "j": ("fugashi", "jaconv", "mojimoji", "pyopenjtalk", "unidic-lite"),
    "z": ("cn2an", "jieba", "ordered-set", "pypinyin", "pypinyin-dict"),
}

# Colab's huggingface_hub otherwise waits on the notebook-UI secret store.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")


def nzap(event, text, **fields):
    """One NZAP app event; `text/plain` is what the console shows."""
    payload = {"v": 1, "app": APP, "event": event, **fields}
    display({"application/vnd.nzap.app+json": payload, "text/plain": text}, raw=True)


def stage(stage_id, label):
    nzap("stage", f"[nzap] {label}…", id=stage_id, label=label)


def pip(*packages):
    # Kokoro and misaki declare Python < 3.13 but run fine on Colab's newer Python.
    command = [sys.executable, "-m", "pip", "install", "-q", "--ignore-requires-python", *packages]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"pip install {' '.join(packages)} failed:\n{result.stderr[-2000:]}")


started = time.time()
state = globals().setdefault("_nzap_apps", {}).setdefault(APP, {})
lang = params.get("language", "a")
warm = "model" in state

if not warm:
    stage("install", "Installing Kokoro and espeak-ng")
    subprocess.run(
        ["apt-get", "-qq", "-y", "install", "espeak-ng"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    # Without --no-deps, misaki[en] drags in spacy-curated-transformers, whose
    # tokenizer has no wheel for Colab's Python and fails to build.
    pip("--no-deps", "kokoro==0.9.4", "misaki==0.9.4")
    pip(*BASE_DEPS)

    stage("load", "Loading Kokoro-82M")
    import torch
    from kokoro import KModel

    state["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    state["model"] = KModel(repo_id="hexgrad/Kokoro-82M").to(state["device"]).eval()
    state["pipelines"] = {}

pipelines = state["pipelines"]
if lang not in pipelines:
    if lang in EXTRAS:
        stage("install", "Installing the phonemiser for this language")
        pip(*EXTRAS[lang])
    stage("load", "Preparing the voice pipeline")
    from kokoro import KPipeline

    pipelines[lang] = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", model=state["model"])

setup_seconds = time.time() - started
nzap("ready", "[nzap] Model ready.", warm=warm, setupSeconds=round(setup_seconds, 2), device=state["device"])

import numpy as np
import soundfile as sf
import torch

text = params["text"].strip()
voice = params.get("voice", "af_heart")
if not voice.startswith(lang):
    raise ValueError(f"Voice {voice} does not speak the selected language ({lang}).")

stage("run", "Synthesising speech")
run_started = time.time()
chunks = []
with torch.inference_mode():
    for result in pipelines[lang](text, voice=voice, speed=float(params.get("speed", 1.0)), split_pattern=r"\n+"):
        if result.audio is not None:
            chunks.append(result.audio.cpu().numpy())
if not chunks:
    raise RuntimeError("Kokoro produced no audio for that text.")

audio = np.concatenate(chunks)
os.makedirs(OUT_DIR, exist_ok=True)
path = f"{OUT_DIR}/{time.strftime('%Y%m%d-%H%M%S')}-{voice}.wav"
sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
run_seconds = time.time() - run_started
duration = len(audio) / SAMPLE_RATE

nzap(
    "output",
    f"[nzap] Saved {path} ({duration:.1f}s of audio)",
    id="speech",
    kind="audio",
    path=path,
    mime="audio/wav",
    meta={"duration": round(duration, 2), "sampleRate": SAMPLE_RATE, "voice": voice},
)
nzap(
    "done",
    f"[nzap] Done in {time.time() - started:.1f}s (audio {duration:.1f}s, {duration / max(run_seconds, 1e-6):.0f}x realtime).",
    seconds={"setup": round(setup_seconds, 2), "run": round(run_seconds, 2)},
    warm=warm,
)
