# NZAP Engine injects a `params` dict before this script runs.
# Image background remover, packaged as an NZAP app (see APPS.md).
#
# A segmentation model (BiRefNet or ISNet, run with ONNX Runtime through rembg)
# predicts a soft mask of the main subject; the mask becomes the alpha channel,
# optionally refined with closed-form alpha matting, and the subject is put on a
# transparent, solid-colour or blurred background. The first run on a runtime
# installs rembg and downloads the chosen model; the model session stays in the
# kernel, so later runs only predict. Results are written to
# /content/nzap/outputs/background-remover/.

import os
import subprocess
import sys
import time

from IPython.display import display

APP = "background-remover"
OUT_DIR = f"/content/nzap/outputs/{APP}"
HOME = f"/content/nzap/apps/{APP}"  # model files, kept across kernel restarts
# rembg pins each model's download URL (its GitHub release assets) and MD5, so
# pinning rembg pins the weights too.
REMBG = "2.0.85"
PYMATTING = "1.1.16"
ONNXRUNTIME = "1.31.0"
MODELS = {  # rembg session name -> (label, download size)
    "birefnet-general-lite": ("BiRefNet lite", "220 MB"),
    "isnet-general-use": ("ISNet", "180 MB"),
    "birefnet-general": ("BiRefNet", "970 MB"),
    "birefnet-portrait": ("BiRefNet portrait", "970 MB"),
}
MAX_SIDE = 6000  # larger photos are scaled down first (36 MP is plenty for a cut-out)

os.environ.setdefault("REMBG_HOME", HOME)


def nzap(event, text, **fields):
    """One NZAP app event; `text/plain` is what the console shows."""
    payload = {"v": 1, "app": APP, "event": event, **fields}
    display({"application/vnd.nzap.app+json": payload, "text/plain": text}, raw=True)


def stage(stage_id, label, progress=None):
    nzap("stage", f"[nzap] {label}…", id=stage_id, label=label, progress=progress)


def pip(*packages):
    command = [sys.executable, "-m", "pip", "install", "-q", *packages]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"pip install {' '.join(packages)} failed:\n{result.stderr[-2000:]}")


def parse_color(value):
    text = (value or "").strip().lstrip("#")
    if len(text) == 3:
        text = "".join(c * 2 for c in text)
    try:
        if len(text) != 6:
            raise ValueError
        return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        raise ValueError(f"Color {value!r} is not a hex colour like #1E90FF.") from None


def load_image(path):
    """The upload as an upright RGB image, or a one-sentence error."""
    from PIL import Image, ImageOps

    if not path or not os.path.isfile(path):
        raise FileNotFoundError("Choose an image to remove the background from.")
    try:
        with Image.open(path) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened)  # phone photos store rotation in EXIF
    except Image.DecompressionBombError:
        raise ValueError("That image is too large to process (over about 180 megapixels).") from None
    except Exception:
        raise ValueError("That file is not a readable PNG, JPEG or WebP image.") from None
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        # An already transparent input: flatten it on white so the model sees what a viewer sees.
        rgba = image.convert("RGBA")
        image = Image.new("RGB", rgba.size, (255, 255, 255))
        image.paste(rgba, mask=rgba.getchannel("A"))
    else:
        image = image.convert("RGB")
    if max(image.size) > MAX_SIDE:
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    return image


def blurred_background(image, mask):
    """A portrait-mode backdrop: the photo blurred using background pixels only, so the
    subject does not smear a halo around itself. Blurring is low-frequency, so it runs
    at a quarter of the size: blur(rgb * bg) / blur(bg), then scale back up."""
    import numpy as np
    from PIL import Image
    from scipy.ndimage import gaussian_filter

    small = (max(1, image.width // 4), max(1, image.height // 4))
    rgb = np.asarray(image.resize(small, Image.Resampling.BOX), dtype=np.float32)
    bg = 1.0 - np.asarray(mask.resize(small, Image.Resampling.BOX), dtype=np.float32) / 255.0
    sigma = max(2.0, max(small) / 80)
    weight = gaussian_filter(bg, sigma)
    blurred = np.stack([gaussian_filter(rgb[..., c] * bg, sigma) for c in range(3)], axis=-1)
    blurred = blurred / np.maximum(weight, 1e-3)[..., None]
    # Where the subject fills a wide area there is no background nearby: fall back to a plain blur.
    plain = np.stack([gaussian_filter(rgb[..., c], sigma) for c in range(3)], axis=-1)
    trust = np.clip(weight / 0.05, 0, 1)[..., None]
    out = blurred * trust + plain * (1 - trust)
    out = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    return out.resize(image.size, Image.Resampling.BICUBIC)


def load_session(name):
    import onnxruntime as ort
    from rembg import new_session
    from rembg.sessions import sessions

    stage("download", f"Downloading {MODELS[name][0]} ({MODELS[name][1]})")
    sessions[name].download_models()  # no-op when the file is already on this runtime
    stage("load", f"Loading {MODELS[name][0]}")
    options = ort.SessionOptions()
    # Use every vCPU (ONNX Runtime defaults to physical cores: one on a Colab CPU VM).
    options.intra_op_num_threads = os.cpu_count() or 2
    # The default memory arena keeps every 1024x1024 activation alive: BiRefNet then
    # tops 12 GB and the kernel is killed. Without it, the peak stays near 7 to 8 GB.
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    return new_session(name, sess_opts=options, providers=["CPUExecutionProvider"])


started = time.time()
model = params.get("model", "birefnet-general-lite")
if model not in MODELS:
    raise ValueError(f"Unknown model {model!r}; pick one of {', '.join(MODELS)}.")
background = params.get("background", "transparent")
alpha_matting = bool(params.get("alpha_matting", False))
crop = bool(params.get("crop_to_subject", False))
fill = parse_color(params.get("color", "#1E90FF")) if background == "color" else None
image = load_image(params.get("image"))  # before any setup, so a bad upload fails at once

state = globals().setdefault("_nzap_apps", {}).setdefault(APP, {})
sessions_loaded = state.setdefault("sessions", {})
warm = model in sessions_loaded

if not warm:
    if "installed" not in state:
        stage("install", "Installing rembg and ONNX Runtime")
        # --no-deps: rembg's own requirements would upgrade Colab's numpy, Pillow,
        # numba and scikit-image under the running kernel. Everything else rembg and
        # pymatting import (scipy, scikit-image, numba, pooch, jsonschema, tqdm) is
        # already on Colab.
        pip("--no-deps", f"rembg=={REMBG}", f"pymatting=={PYMATTING}")
        pip(f"onnxruntime=={ONNXRUNTIME}")
        state["installed"] = True
    sessions_loaded[model] = load_session(model)

setup_seconds = time.time() - started
nzap("ready", f"[nzap] {MODELS[model][0]} ready.", warm=warm, setupSeconds=round(setup_seconds, 2), device="CPU")

import numpy as np
from PIL import Image

run_started = time.time()

stage("run", f"Finding the subject with {MODELS[model][0]}")
mask = sessions_loaded[model].predict(image)[0].convert("L")

cutout = None
if alpha_matting:
    from rembg.bg import alpha_matting_cutout

    stage("run", "Refining the edges (alpha matting)")
    try:
        # A trimap from the mask (sure subject > 240, sure background < 10, both eroded
        # by a size that scales with the photo), solved for the alpha in between; the
        # subject's colours are unmixed from the old background at the same time.
        erode = max(10, round(max(image.size) / 200))
        cutout = alpha_matting_cutout(image, mask, 240, 10, erode)
        mask = cutout.getchannel("A")
    except ValueError:
        print("[nzap] Alpha matting did not converge on this image; keeping the model's mask.")
if cutout is None:
    cutout = image.convert("RGBA")
    cutout.putalpha(mask)

alpha = np.asarray(mask)
if alpha.max() < 16:
    raise RuntimeError("No subject was found in this image. Try another model or a photo with a clear subject.")

if crop:
    rows, cols = np.nonzero(alpha > 16)
    pad = round(0.02 * max(image.size))
    box = (max(0, cols.min() - pad), max(0, rows.min() - pad),
           min(image.width, cols.max() + 1 + pad), min(image.height, rows.max() + 1 + pad))
    image, mask, cutout = image.crop(box), mask.crop(box), cutout.crop(box)

stage("run", "Compositing")
if background == "transparent":
    result = cutout
elif background == "blur":
    result = blurred_background(image, mask).convert("RGBA")
    result.alpha_composite(cutout)
    result = result.convert("RGB")
else:
    colour = {"white": (255, 255, 255), "black": (0, 0, 0)}.get(background, fill)
    if colour is None:
        raise ValueError(f"Unknown background {background!r}.")
    result = Image.new("RGBA", cutout.size, colour + (255,))
    result.alpha_composite(cutout)
    result = result.convert("RGB")

os.makedirs(OUT_DIR, exist_ok=True)
stem = os.path.splitext(os.path.basename(params.get("image") or "image"))[0][:60] or "image"
stamp = time.strftime("%Y%m%d-%H%M%S")
path = f"{OUT_DIR}/{stamp}-{stem}-{background}.png"
mask_path = f"{OUT_DIR}/{stamp}-{stem}-mask.png"
result.save(path, optimize=False, compress_level=6)
mask.save(mask_path)
run_seconds = time.time() - run_started
coverage = float((alpha > 127).mean())

nzap(
    "output",
    f"[nzap] Saved {path}",
    id="image",
    kind="image",
    path=path,
    mime="image/png",
    meta={"width": result.width, "height": result.height, "model": model, "background": background,
          "alphaMatting": alpha_matting, "cropped": crop},
)
nzap(
    "output",
    f"[nzap] Saved {mask_path}",
    id="mask",
    kind="image",
    path=mask_path,
    mime="image/png",
    meta={"width": mask.width, "height": mask.height, "subjectCoverage": round(coverage, 4)},
)
nzap(
    "done",
    f"[nzap] Done in {time.time() - started:.1f}s ({result.width}x{result.height}, "
    f"subject covers {coverage:.0%} of the frame).",
    seconds={"setup": round(setup_seconds, 2), "run": round(run_seconds, 2)},
    warm=warm,
)
