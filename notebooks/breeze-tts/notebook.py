# NZAP Engine injects a `params` dict before this script runs.
# Breeze TTS 2 (3B) expressive speech, packaged as an NZAP app (see APPS.md).
#
# Describe a voice in plain words (voice design), or upload a short reference
# clip with its transcript (voice cloning). The first run on a runtime
# downloads the 7.7 GB checkpoint and loads it onto the GPU (about 8 GB of
# VRAM, so a T4 or better); later runs on the same runtime only generate.
# Audio is written to /content/nzap/outputs/breeze-tts/.

import os
import subprocess
import sys
import time

from IPython.display import display

APP = "breeze-tts"
OUT_DIR = f"/content/nzap/outputs/{APP}"
HOME = f"/content/nzap/apps/{APP}"
# Inference code, pinned to a reviewed commit of github.com/breezeblue-ai/breeze-tts.
REPO = "https://github.com/breezeblue-ai/breeze-tts.git"
COMMIT = "58ec70ce5fa4cc361bdebf77ec40d1365da00ab2"
MODEL = "BreezeBlue/Breeze-TTS-2"
MODEL_REVISION = "3e28c5151381a722f1d8661b4118c298caa77aa4"
MAX_NEW_TOKENS = 1500
MAX_SEQ_LEN = 2048


def nzap(event, text, **fields):
    """One NZAP app event; `text/plain` is what the console shows."""
    payload = {"v": 1, "app": APP, "event": event, **fields}
    display({"application/vnd.nzap.app+json": payload, "text/plain": text}, raw=True)


def stage(stage_id, label, progress=None):
    nzap("stage", f"[nzap] {label}…", id=stage_id, label=label, progress=progress)


def run(*command, **kwargs):
    subprocess.run(list(command), check=True, **kwargs)


started = time.time()
state = globals().setdefault("_nzap_apps", {}).setdefault(APP, {})
warm = "runtime" in state

if not warm:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Breeze TTS 2 needs a GPU runtime (T4 or better). Switch the runtime to a GPU.")

    stage("install", "Installing Breeze TTS 2")
    code_dir = f"{HOME}/src"
    if not os.path.isdir(f"{code_dir}/.git"):
        run("git", "clone", "-q", "--filter=blob:none", REPO, code_dir)
    run("git", "-C", code_dir, "checkout", "-q", COMMIT)
    run(
        sys.executable, "-m", "pip", "install", "-q",
        "transformers==4.57.3", "accelerate==1.12.0", "soundfile", "librosa", "einops", "onnxruntime", "sox",
    )
    run(sys.executable, "-m", "pip", "install", "-q", "--no-deps", "qwen-tts==0.1.1")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    stage("download", "Downloading the 7.7 GB checkpoint")
    from huggingface_hub import snapshot_download

    weights = snapshot_download(MODEL, revision=MODEL_REVISION, local_dir=f"{HOME}/weights")

    stage("load", "Loading Breeze TTS 2 onto the GPU")
    import transformers

    if transformers.__version__ != "4.57.3":
        raise RuntimeError(
            f"This runtime already imported transformers {transformers.__version__}. "
            "Restart the runtime, then run Breeze TTS 2 again."
        )
    from pathlib import Path

    from breeze_infer.runtime import load_runtime, update_generation_config_for_breeze
    from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

    tokenizer, model, audio_tokenizer = load_runtime(Path(weights), device="cuda:0", attn_implementation="sdpa")
    update_generation_config_for_breeze(model)
    state["tokenizer"] = tokenizer
    state["audio_tokenizer"] = audio_tokenizer
    state["model"] = model
    state["runtime"] = FastBreezeStreamingRuntime(
        model,
        audio_tokenizer,
        FastStreamingConfig(max_new_tokens=MAX_NEW_TOKENS, max_seq_len=MAX_SEQ_LEN, repetition_penalty=1.1),
        tokenizer=tokenizer,
    )
    state["gpu"] = torch.cuda.get_device_name(0)

setup_seconds = time.time() - started
nzap("ready", "[nzap] Model ready.", warm=warm, setupSeconds=round(setup_seconds, 2), device=state["gpu"])

import numpy as np
import soundfile as sf
from breeze_infer.runtime import set_all_seeds
from breeze_infer.templates import get_template, prepare_inputs, select_template_name

runtime = state["runtime"]
paragraphs = [part.strip() for part in params["text"].split("\n\n") if part.strip()]
if not paragraphs:
    raise ValueError("Enter some text to speak.")
instruction = (params.get("instruction") or "").strip()
ref_audio = (params.get("reference_audio") or "").strip()
ref_text = (params.get("reference_text") or "").strip()
if bool(ref_audio) != bool(ref_text):
    raise ValueError("A reference clip needs its exact transcript (and the other way round).")
if ref_audio and not os.path.isfile(ref_audio):
    raise FileNotFoundError(f"Reference audio not found on the runtime: {ref_audio}")
seed = int(params.get("seed", 42))
cfg_scale = float(params.get("cfg_scale", 4.0))
keep_voice = bool(params.get("consistent_voice", True))


def speak(text, ref_path, ref_transcript):
    request = {"id": "nzap", "text": text, "speaker": "S0"}
    if instruction:
        request["instruction"] = instruction
    if ref_path:
        request["ref_audio_path"] = ref_path
        request["ref_text"] = ref_transcript
    set_all_seeds(seed)
    inputs = prepare_inputs(
        state["tokenizer"],
        state["audio_tokenizer"],
        state["model"],
        [request],
        get_template(select_template_name(request)),
        # Classifier-free guidance steers the instruction; plain cloning has none.
        guidance_scale=cfg_scale if instruction else 1.0,
        guidance_scale_ref=None,
        guidance_scale_ins=None,
    )
    chunks = [chunk.audio for chunk in runtime.iter_audio_chunks(inputs, request_id="nzap", seed=seed)]
    if not chunks:
        raise RuntimeError("Breeze produced no audio for that text.")
    return np.concatenate(chunks).astype(np.float32)


os.makedirs(OUT_DIR, exist_ok=True)
stamp = time.strftime("%Y%m%d-%H%M%S")
pieces = []
run_started = time.time()
for index, paragraph in enumerate(paragraphs):
    stage("run", f"Generating paragraph {index + 1} of {len(paragraphs)}", progress=index / len(paragraphs))
    audio = speak(paragraph, ref_audio, ref_text)
    pieces.append(audio)
    # Without a reference clip, later paragraphs clone the first one so the
    # designed voice stays the same across the whole text.
    if index == 0 and keep_voice and not ref_audio and len(paragraphs) > 1:
        ref_audio = f"{OUT_DIR}/{stamp}-voice.wav"
        sf.write(ref_audio, audio, runtime.sample_rate, subtype="PCM_16")
        ref_text = paragraph

pause = np.zeros(int(runtime.sample_rate * 0.45), dtype=np.float32)
audio = np.concatenate([part for piece in pieces for part in (piece, pause)][:-1])
# Where each paragraph starts and ends, for captions and syncing.
segments, offset = [], 0
for paragraph, piece in zip(paragraphs, pieces):
    segments.append({"start": round(offset / runtime.sample_rate, 3), "end": round((offset + len(piece)) / runtime.sample_rate, 3), "text": paragraph})
    offset += len(piece) + len(pause)
path = f"{OUT_DIR}/{stamp}.wav"
sf.write(path, audio, runtime.sample_rate, subtype="PCM_16")
run_seconds = time.time() - run_started
duration = len(audio) / runtime.sample_rate

nzap(
    "output",
    f"[nzap] Saved {path} ({duration:.1f}s of audio)",
    id="speech",
    kind="audio",
    path=path,
    mime="audio/wav",
    meta={"duration": round(duration, 2), "sampleRate": runtime.sample_rate, "segments": segments},
)
nzap(
    "done",
    f"[nzap] Done in {time.time() - started:.1f}s (audio {duration:.1f}s, {duration / max(run_seconds, 1e-6):.2f}x realtime).",
    seconds={"setup": round(setup_seconds, 2), "run": round(run_seconds, 2)},
    warm=warm,
)
