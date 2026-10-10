# NZAP Engine injects a `params` dict before this script runs.
# Speech to text and subtitles with faster-whisper, packaged as an NZAP app
# (see APPS.md).
#
# faster-whisper runs OpenAI's Whisper models on CTranslate2: int8 on a CPU
# runtime, float16 on a GPU. The first run installs it and loads the chosen
# model; models stay in the kernel (one per size), so later runs on the same
# runtime go straight to transcription. The transcript and the .srt, .vtt and
# .txt files are written to /content/nzap/outputs/speech-to-text/.

import os
import re
import subprocess
import sys
import time

from IPython.display import display

APP = "speech-to-text"
OUT_DIR = f"/content/nzap/outputs/{APP}"
SAMPLE_RATE = 16_000  # what Whisper listens at
PACKAGES = ("faster-whisper==1.2.1", "ctranslate2==4.8.2")
# CTranslate2 conversions of the Whisper weights (MIT), pinned to a commit.
MODELS = {
    "tiny": ("Systran/faster-whisper-tiny", "d90ca5fe260221311c53c58e660288d3deb8d356"),
    "base": ("Systran/faster-whisper-base", "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"),
    "small": ("Systran/faster-whisper-small", "536b0662742c02347bc0e980a01041f333bce120"),
    "medium": ("Systran/faster-whisper-medium", "08e178d48790749d25932bbc082711ddcfdfbc4f"),
    "large-v3-turbo": ("dropbox-dash/faster-whisper-large-v3-turbo", "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"),
}
# Subtitle cues built from word timestamps stay short enough to read.
CUE_CHARS, CUE_SECONDS = 42, 6.0

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


def decode(path):
    """Any audio or video file -> 16 kHz mono float32, decoded by ffmpeg."""
    import numpy as np

    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    result = subprocess.run(command, capture_output=True)
    if result.returncode:
        reason = result.stderr.decode(errors="replace").strip().splitlines()[-1:] or ["unknown error"]
        raise RuntimeError(f"ffmpeg could not read {os.path.basename(path)}: {reason[0]}")
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size < SAMPLE_RATE // 10:
        raise RuntimeError(f"{os.path.basename(path)} has no audio track to transcribe.")
    return audio


def clock(seconds, sep=","):
    """12.5 -> 00:00:12,500 (SubRip) or 00:00:12.500 (WebVTT)."""
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def short(seconds):
    """12.5 -> 0:12.5, for the table."""
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:04.1f}" if h else f"{m}:{s:04.1f}"


def cues_from_words(words):
    """Regroup word timestamps into subtitle-sized cues (≤ 42 characters, ≤ 6 s)."""
    cues, current = [], []
    for word in words:
        if current:
            text = "".join(w.word for w in current + [word]).strip()
            if len(text) > CUE_CHARS or word.end - current[0].start > CUE_SECONDS:
                cues.append(current)
                current = []
        current.append(word)
        # Close the cue at the end of a sentence once it has a few words.
        if word.word.strip()[-1:] in ".?!" and len(current) >= 3:
            cues.append(current)
            current = []
    if current:
        cues.append(current)
    return [(c[0].start, c[-1].end, "".join(w.word for w in c).strip()) for c in cues]


started = time.time()
state = globals().setdefault("_nzap_apps", {}).setdefault(APP, {})
models = state.setdefault("models", {})
size = params.get("model", "small")
if size not in MODELS:
    raise RuntimeError(f"Unknown model {size!r}; pick one of {', '.join(MODELS)}.")
warm = size in models

source = params.get("audio", "")
if not source or not os.path.isfile(source):
    raise RuntimeError("Upload an audio or video file to transcribe.")

if not warm:
    if not state.get("installed"):
        stage("install", "Installing faster-whisper")
        pip(*PACKAGES)
        state["installed"] = True

    import ctranslate2
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download

    repo, revision = MODELS[size]
    stage("download", f"Downloading Whisper {size}")
    model_dir = snapshot_download(repo, revision=revision)

    stage("load", f"Loading Whisper {size}")
    threads = os.cpu_count() or 2
    model, device = None, "cpu"
    if ctranslate2.get_cuda_device_count() > 0:
        try:
            model, device = WhisperModel(model_dir, device="cuda", compute_type="float16"), "cuda"
        except Exception as error:  # e.g. a CUDA / cuDNN version this CTranslate2 build cannot use
            print(f"GPU load failed ({error}); using the CPU instead.")
    if model is None:
        model = WhisperModel(model_dir, device="cpu", compute_type="int8", cpu_threads=threads)
    models[size] = {"model": model, "device": device}

model, device = models[size]["model"], models[size]["device"]
setup_seconds = time.time() - started
nzap("ready", f"[nzap] Whisper {size} ready on {device}.", warm=warm, setupSeconds=round(setup_seconds, 2), device=device)

run_started = time.time()
stage("run", "Reading the audio", 0)
audio = decode(source)
duration = len(audio) / SAMPLE_RATE

language = params.get("language", "auto")
task = params.get("task", "transcribe")
word_timestamps = bool(params.get("word_timestamps", False))
segments_iter, info = model.transcribe(
    audio,
    language=None if language == "auto" else language,
    task=task,
    beam_size=5,
    vad_filter=bool(params.get("vad_filter", True)),
    word_timestamps=word_timestamps,
)

# Segments stream in as Whisper decodes; report progress against the clip length.
segments, last_report = [], 0.0
for segment in segments_iter:
    segments.append(segment)
    if time.time() - last_report > 1.0:
        last_report = time.time()
        stage("run", f"Transcribing {short(min(segment.end, duration))} of {short(duration)}", min(segment.end / duration, 0.99))
if not segments:
    raise RuntimeError("Whisper heard no speech in that file.")

# Subtitle cues: Whisper's own segments, or word-timed cues when asked for.
if word_timestamps:
    cues = [cue for s in segments if s.words for cue in cues_from_words(s.words)]
else:
    cues = [(s.start, s.end, s.text.strip()) for s in segments]
cues = [(start, max(end, start + 0.2), text) for start, end, text in cues if text]

transcript = "\n".join(s.text.strip() for s in segments if s.text.strip())
srt = "\n".join(f"{i}\n{clock(a)} --> {clock(b)}\n{text}\n" for i, (a, b, text) in enumerate(cues, 1))
vtt = "WEBVTT\n\n" + "\n".join(f"{clock(a, '.')} --> {clock(b, '.')}\n{text}\n" for a, b, text in cues)

os.makedirs(OUT_DIR, exist_ok=True)
name = re.sub(r"[^A-Za-z0-9_-]+", "-", os.path.splitext(os.path.basename(source))[0]).strip("-")[:60] or "audio"
base = f"{OUT_DIR}/{time.strftime('%Y%m%d-%H%M%S')}-{name}"
for ext, body in (("txt", transcript + "\n"), ("srt", srt), ("vtt", vtt)):
    with open(f"{base}.{ext}", "w", encoding="utf-8") as file:
        file.write(body)
run_seconds = time.time() - run_started

detected = f"{info.language} ({info.language_probability:.0%} sure)" if language == "auto" else info.language
print(f"Language: {detected}\n\n{transcript}")
nzap("output", f"[nzap] Transcript ({len(transcript.split())} words, language {detected})", id="transcript", kind="text", text=transcript)
nzap(
    "output",
    f"[nzap] {len(segments)} segments",
    id="segments",
    kind="table",
    columns=["Start", "End", "Text"],
    rows=[[short(s.start), short(s.end), s.text.strip()] for s in segments],
)
for ext, mime in (("srt", "application/x-subrip"), ("vtt", "text/vtt"), ("txt", "text/plain")):
    path = f"{base}.{ext}"
    nzap("output", f"[nzap] Saved {path}", id=ext, kind="file", path=path, mime=mime, filename=os.path.basename(path))
nzap(
    "done",
    f"[nzap] Done in {time.time() - started:.1f}s ({duration:.0f}s of audio, {duration / max(run_seconds, 1e-6):.1f}x realtime).",
    seconds={"setup": round(setup_seconds, 2), "run": round(run_seconds, 2)},
    warm=warm,
)
