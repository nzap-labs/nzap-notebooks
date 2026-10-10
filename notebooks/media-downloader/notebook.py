# NZAP Engine injects a `params` dict before this script runs.
# Downloads a video or its audio with yt-dlp (YouTube and the ~1,800 other
# sites it supports), packaged as an NZAP app (see APPS.md).
#
# The first run on a runtime installs a pinned yt-dlp; later runs reuse it.
# Videos come out as MP4 with H.264 video and AAC audio, so they play in any
# browser; audio comes out in the format you pick. Files are written to
# /content/nzap/outputs/media-downloader/. Only download content you have the
# right to download.

import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

from IPython.display import display

APP = "media-downloader"
OUT_DIR = f"/content/nzap/outputs/{APP}"
# yt-dlp is Unlicense. YouTube hides its formats behind a JavaScript challenge:
# yt-dlp-ejs ships the solver (so nothing is fetched at run time) and Deno runs
# it (Colab's Node.js 20 is older than the 22 yt-dlp needs). These are the
# versions yt-dlp's own `pin` / `pin-deno` extras use.
YT_DLP = "2026.8.19"
PINS = {"yt-dlp": YT_DLP, "yt-dlp-ejs": "0.8.0", "deno": "2.9.5"}
HEIGHTS = {"1080p": 1080, "720p": 720, "480p": 480, "360p": 360}
AUDIO_MIME = {"mp3": "audio/mpeg", "m4a": "audio/mp4", "opus": "audio/ogg", "wav": "audio/wav", "flac": "audio/flac"}
LOSSY = {"mp3", "m4a", "opus"}


def nzap(event, text, **fields):
    """One NZAP app event; `text/plain` is what the console shows."""
    payload = {"v": 1, "app": APP, "event": event, **fields}
    display({"application/vnd.nzap.app+json": payload, "text/plain": text}, raw=True)


def stage(stage_id, label, progress=None):
    extra = {} if progress is None else {"progress": round(progress, 3)}
    nzap("stage", f"[nzap] {label}…", id=stage_id, label=label, **extra)


def pip(*packages):
    command = [sys.executable, "-m", "pip", "install", "-q", *packages]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"pip install {' '.join(packages)} failed:\n{result.stderr[-2000:]}")


def installed(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def seconds(value, name):
    """'90', '1:30', '1:02:03' or '' (unset) -> seconds."""
    value = str(value or "").strip()
    if not value:
        return None
    if not re.fullmatch(r"\d+(?:\.\d+)?|(?:\d+:){1,2}\d+(?:\.\d+)?", value):
        raise ValueError(f"{name} must look like 90, 1:30 or 1:02:03 (got {value!r}).")
    total = 0.0
    for part in value.split(":"):
        total = total * 60 + float(part)
    return total


def clock(total):
    total = int(round(total or 0))
    hours, rest = divmod(total, 3600)
    return f"{hours}:{rest // 60:02d}:{rest % 60:02d}" if hours else f"{rest // 60}:{rest % 60:02d}"


def human_size(size):
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024


def probe(path):
    """Codecs, size and duration of a finished file, from ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
        capture_output=True, text=True,
    )
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    return {
        "video": video and video.get("codec_name"),
        "audio": audio and audio.get("codec_name"),
        "width": video and video.get("width"),
        "height": video and video.get("height"),
        "sampleRate": audio and int(audio.get("sample_rate") or 0),
        "duration": float(data.get("format", {}).get("duration") or 0),
    }


def browser_mp4(path):
    """Re-encode to H.264/AAC MP4 unless the download already is (some sites only serve VP9/AV1)."""
    info = probe(path)
    if path.endswith(".mp4") and info["video"] == "h264" and info["audio"] in ("aac", None):
        return path, info
    stage("convert", "Converting to H.264 MP4")
    target = os.path.splitext(path)[0] + ".h264.mp4"
    command = ["ffmpeg", "-y", "-v", "error", "-i", path, "-map", "0:v:0", "-map", "0:a:0?",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", target]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"ffmpeg could not convert the video to H.264: {result.stderr.strip()[-300:]}")
    os.remove(path)
    final = target[: -len(".h264.mp4")] + ".mp4"
    os.replace(target, final)
    return final, probe(final)


def friendly(error):
    """Turn a yt-dlp error into the one sentence the app shows."""
    message = re.sub(r"\x1b\[[0-9;]*m", "", str(error))
    lower = message.lower()
    if "confirm you" in lower and "bot" in lower:
        return ("YouTube is blocking downloads from this Colab runtime's IP address (\"Sign in to confirm you're not "
                "a bot\"); try again later or on a new runtime, or use a link from another site.")
    if "logged-in" in lower or "login required" in lower or "--cookies" in lower or "account credentials" in lower:
        return "That site only allows signed-in downloads for this link, and this app does not sign in for you."
    if "unsupported url" in lower:
        return "yt-dlp does not recognise that link; paste the address of a video page or a direct media URL."
    if "private video" in lower or "members-only" in lower or "join this channel" in lower:
        return "That video is private or members-only, so it cannot be downloaded without signing in."
    if "confirm your age" in lower or "age-restricted" in lower or "inappropriate for some users" in lower:
        return "That video is age-restricted, so it cannot be downloaded without signing in."
    if "not available in your country" in lower or "geo restrict" in lower or "geo-restrict" in lower:
        return "That video is not available in the country this Colab runtime is in."
    if "live event will begin" in lower or "premieres in" in lower or "is_upcoming" in lower:
        return "That video has not started yet (an upcoming live stream or premiere)."
    if "http error 429" in lower or "too many requests" in lower:
        return "The site is rate-limiting this Colab runtime (HTTP 429); wait a few minutes and try again."
    if "http error 403" in lower:
        return "The site refused the download (HTTP 403 Forbidden); it may block Colab, or the link may have expired."
    if "http error 404" in lower or "video unavailable" in lower or "video is unavailable" in lower or "does not exist" in lower:
        return "That video is unavailable: it was removed, or the link is wrong."
    if "requested format is not available" in lower:
        return "No downloadable format matches those settings; try quality Best or audio mode."
    if "drm" in lower:
        return "That video is DRM-protected, and yt-dlp cannot download DRM content."
    first = message.replace("ERROR: ", "").strip().splitlines()[0] if message.strip() else "unknown error"
    return f"yt-dlp could not download that link: {first}"


started = time.time()
state = globals().setdefault("_nzap_apps", {}).setdefault(APP, {})
warm = state.get("version") == YT_DLP

if not warm:
    if any(installed(name) != version for name, version in PINS.items()):
        stage("install", f"Installing yt-dlp {YT_DLP}")
        pip(*(f"{name}=={version}" for name, version in PINS.items()))
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool} is missing on this runtime; Colab normally preinstalls it.")
    import yt_dlp

    if [int(n) for n in yt_dlp.version.__version__.split(".")[:3]] != [int(n) for n in YT_DLP.split(".")]:
        # An older yt-dlp was already imported in this kernel; pip cannot swap it in place.
        raise RuntimeError("An older yt-dlp is already loaded in this runtime; restart the runtime and run again.")
    import deno

    state["js_runtimes"] = {"deno": {"path": deno.find_deno_bin()}}
    state["version"] = YT_DLP

import yt_dlp
from yt_dlp.utils import DownloadError

setup_seconds = time.time() - started
nzap("ready", f"[nzap] yt-dlp {YT_DLP} ready.", warm=warm, setupSeconds=round(setup_seconds, 2), device="cpu")

# ── Parameters ───────────────────────────────────────────────────────────────
url = str(params.get("url") or "").strip()
if not re.match(r"^https?://", url):
    raise ValueError("Paste a full link starting with http:// or https://.")
mode = params.get("mode", "video")
quality = params.get("quality", "720p")
audio_format = params.get("audio_format", "mp3")
bitrate = str(params.get("audio_bitrate", "192"))
start = seconds(params.get("start"), "Start")
end = seconds(params.get("end"), "End")
if start is not None and end is not None and end <= start:
    raise ValueError("End must be later than Start.")
playlist = bool(params.get("playlist", False))
max_items = max(1, min(int(params.get("max_items") or 10), 50)) if playlist else 1

if mode == "audio":
    # Best audio-only stream, converted by ffmpeg to the chosen format.
    fmt = "bestaudio/best"
    postprocessors = [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": audio_format,
        **({"preferredquality": bitrate} if audio_format in LOSSY else {}),
    }]
else:
    # Prefer H.264 + AAC (plays everywhere, no re-encode needed), then anything.
    cap = f"[height<={HEIGHTS[quality]}]" if quality in HEIGHTS else ""
    fmt = "/".join([
        f"bv*{cap}[vcodec^=avc1]+ba[acodec^=mp4a]",
        f"b{cap}[vcodec^=avc1][acodec^=mp4a]",
        f"bv*{cap}+ba",
        f"b{cap}",
        "bv*+ba/b",
    ])
    postprocessors = []

progress = {"last": 0.0}


def on_progress(update):
    # Throttled: one progress event every couple of seconds is plenty.
    if update.get("status") != "downloading" or time.time() - progress["last"] < 2:
        return
    progress["last"] = time.time()
    total = update.get("total_bytes") or update.get("total_bytes_estimate")
    done_bytes = update.get("downloaded_bytes") or 0
    info = update.get("info_dict") or {}
    label = f"Downloading {(info.get('title') or 'media')[:60]}"
    if playlist and info.get("playlist_index"):
        label += f" ({info['playlist_index']}/{min(max_items, info.get('n_entries') or max_items)})"
    stage("download", label, min(done_bytes / total, 1.0) if total else None)


skipped = []


def check_start(info, *, incomplete=False):
    # A start past the end would give an empty file; say so instead.
    duration = info.get("duration")
    if start and duration and start >= duration:
        skipped.append(f"Start ({clock(start)}) is past the end of the video ({clock(duration)}).")
        return skipped[-1]
    return None


stamp = time.strftime("%Y%m%d-%H%M%S")
options = {
    "format": fmt,
    "merge_output_format": "mp4",
    "postprocessors": postprocessors,
    "outtmpl": f"{OUT_DIR}/{stamp}-%(title).80B [%(id)s].%(ext)s",
    "windowsfilenames": True,  # names that are also safe once copied to Windows
    "overwrites": True,
    "noplaylist": not playlist,
    "playlistend": max_items,  # also caps a playlist link to its first item when Playlist is off
    "ignoreerrors": "only_download" if playlist else False,  # one bad item should not sink a playlist
    "quiet": True,
    "noprogress": True,
    "progress_hooks": [on_progress],
    "js_runtimes": state["js_runtimes"],
    "retries": 3,
    "socket_timeout": 30,
}
if start is not None or end is not None:
    options["match_filter"] = check_start
    cut = (start or 0, end if end is not None else float("inf"))
    if mode == "video":
        # ffmpeg fetches just that range; an exact cut means re-encoding the clip
        # (fast preset: CPU runtimes have 2 cores).
        options["download_ranges"] = lambda info, ydl: [{"start_time": cut[0], "end_time": cut[1]}]
        options["force_keyframes_at_cuts"] = True
        options["external_downloader_args"] = {"ffmpeg_o": ["-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]}
    else:
        # Audio is small: fetch it whole and cut exactly while converting it.
        options["postprocessor_args"] = {
            "extractaudio": ["-ss", str(cut[0])] + ([] if end is None else ["-to", str(cut[1])])
        }

os.makedirs(OUT_DIR, exist_ok=True)
stage("download", "Looking up the link")
run_started = time.time()
for attempt in range(3):
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
        break
    except DownloadError as error:
        # YouTube now and then refuses a stream URL (HTTP 403); a fresh lookup
        # gets new URLs and usually works.
        if "http error 403" in str(error).lower() and attempt < 2:
            print(f"[nzap] HTTP 403 from the site; retrying ({attempt + 1}/2)…")
            time.sleep(2)
            continue
        raise RuntimeError(friendly(error)) from None

if skipped and not playlist:
    raise ValueError(skipped[0])
if info is None:
    raise RuntimeError("Nothing was downloaded from that link.")
if info.get("entries") is not None and not playlist:
    print("[nzap] That link is a playlist; downloaded only its first item (turn on Playlist for more).")


def flatten(node):
    if node is None:
        return
    if node.get("entries") is not None:
        for child in node["entries"]:
            yield from flatten(child)
    else:
        yield node


# The final path of each download, after merging and audio conversion.
items = []
for entry in flatten(info):
    downloads = entry.get("requested_downloads") or []
    path = downloads[-1].get("filepath") if downloads else None
    if path and os.path.exists(path):
        items.append((entry, path))
if not items:
    raise RuntimeError("Nothing was downloaded; every item failed or was unavailable.")

# ── Results ──────────────────────────────────────────────────────────────────
rows, files = [], []
for result, path in items:
    if mode == "video":
        path, media = browser_mp4(path)
        resolution = f"{media['width']}x{media['height']}" if media["width"] else "-"
        codecs = f"{media['video']} / {media['audio'] or 'no audio'}"
    else:
        media = probe(path)
        resolution = "audio only"
        codecs = f"{media['audio']}" + (f" {bitrate} kbps" if audio_format in LOSSY else "")
    size = os.path.getsize(path)
    files.append((path, media))
    rows.append([
        result.get("title") or os.path.basename(path),
        result.get("uploader") or result.get("channel") or "-",
        clock(media["duration"] or result.get("duration")),
        resolution,
        codecs,
        human_size(size),
    ])
    print(f"[nzap] Saved {path} ({human_size(size)})")

if len(files) == 1:
    path, media = files[0]
    if mode == "video":
        nzap("output", f"[nzap] Video: {path}", id="video", kind="video", path=path, mime="video/mp4")
    else:
        nzap(
            "output",
            f"[nzap] Audio: {path}",
            id="audio",
            kind="audio",
            path=path,
            mime=AUDIO_MIME[audio_format],
            meta={"duration": round(media["duration"], 2), "sampleRate": media["sampleRate"], "format": audio_format},
        )
else:
    stage("run", f"Zipping {len(files)} files")
    archive = f"{OUT_DIR}/{stamp}-playlist.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as bundle:  # media is already compressed
        for path, _ in files:
            bundle.write(path, os.path.basename(path))
    nzap(
        "output",
        f"[nzap] {len(files)} files zipped into {archive} ({human_size(os.path.getsize(archive))})",
        id="archive",
        kind="file",
        path=archive,
        mime="application/zip",
        filename=os.path.basename(archive),
    )

nzap(
    "output",
    f"[nzap] Downloaded {len(rows)} file(s).",
    id="details",
    kind="table",
    columns=["Title", "Uploader", "Duration", "Resolution", "Codecs", "Size"],
    rows=rows,
)
run_seconds = time.time() - run_started
nzap(
    "done",
    f"[nzap] Done in {time.time() - started:.1f}s.",
    seconds={"setup": round(setup_seconds, 2), "run": round(run_seconds, 2)},
    warm=warm,
)
