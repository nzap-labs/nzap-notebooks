# NZAP Engine injects a `params` dict before this script runs.
# AI image enhancer: Real-ESRGAN super-resolution (2x or 4x) with optional GFPGAN
# face restoration, packaged as an NZAP app (see APPS.md). Runs on a CPU runtime
# and uses a GPU automatically when the runtime has one.
#
# Models load through spandrel (MIT), which recognises each architecture from the
# weights alone, so no model code is fetched from Git. Weights come from the
# official GitHub releases (Real-ESRGAN: BSD-3-Clause, GFPGAN: Apache-2.0) and the
# YuNet face detector (MIT) from OpenCV's Hugging Face repo at a fixed revision;
# every file is checked against its SHA-256 before it is loaded. Loaded models stay
# in the kernel, so later runs on the same runtime skip straight to enhancing.
# Results are written to /content/nzap/outputs/image-enhancer/.

import math
import os
import subprocess
import sys
import time

from IPython.display import display

APP = "image-enhancer"
OUT_DIR = f"/content/nzap/outputs/{APP}"
WEIGHTS_DIR = f"/content/nzap/apps/{APP}/weights"
SPANDREL = "0.4.2"
ESRGAN = "https://github.com/xinntao/Real-ESRGAN/releases/download"

# file name -> (url, sha256, size in MB)
WEIGHTS = {
    "realesr-general-x4v3.pth": (
        f"{ESRGAN}/v0.2.5.0/realesr-general-x4v3.pth",
        "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292", 5),
    "RealESRGAN_x4plus.pth": (
        f"{ESRGAN}/v0.1.0/RealESRGAN_x4plus.pth",
        "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1", 67),
    "RealESRGAN_x2plus.pth": (
        f"{ESRGAN}/v0.2.1/RealESRGAN_x2plus.pth",
        "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb", 67),
    "RealESRGAN_x4plus_anime_6B.pth": (
        f"{ESRGAN}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da", 18),
    "GFPGANv1.4.pth": (
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad", 349),
    "face_detection_yunet_2023mar.onnx": (
        "https://huggingface.co/opencv/face_detection_yunet/resolve/"
        "3cc26e7f1014a5ee5d74a42acee58bafc9d0a310/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4", 0.2),
}
# The `model` option -> weights per output scale. A scale without its own weights runs
# the 4x model and resizes the result with Lanczos. RealESRGAN_x2plus works at half the
# resolution internally, so 2x photos take about a quarter of the 4x time.
MODELS = {
    "fast": {4: "realesr-general-x4v3.pth"},
    "photo": {4: "RealESRGAN_x4plus.pth", 2: "RealESRGAN_x2plus.pth"},
    "anime": {4: "RealESRGAN_x4plus_anime_6B.pth"},
}
TILE_PAD = 10  # context pixels around every tile, cropped off afterwards (no seams)
MAX_FACES = 20
# Where the eyes, nose tip and mouth corners sit in GFPGAN's 512x512 aligned face
# (the FFHQ template from facexlib, which GFPGAN was trained with).
FACE_TEMPLATE = [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                 [201.26117, 371.41043], [313.08905, 371.15118]]


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


def fetch(name):
    """Download one weights file (once per runtime), verified against its SHA-256."""
    import hashlib
    import urllib.request

    path = f"{WEIGHTS_DIR}/{name}"
    if os.path.exists(path):
        return path
    url, sha256, size_mb = WEIGHTS[name]
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    label = f"Downloading {name} ({size_mb:g} MB)"
    stage("download", label, progress=0.0)
    digest, done, last = hashlib.sha256(), 0, 0.0
    request = urllib.request.Request(url, headers={"User-Agent": "nzap-image-enhancer"})
    with urllib.request.urlopen(request, timeout=120) as response, open(path + ".part", "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        while chunk := response.read(1 << 20):
            out.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if total and time.time() - last > 2:
                stage("download", label, progress=round(done / total, 3))
                last = time.time()
    if digest.hexdigest() != sha256:
        os.remove(path + ".part")
        raise RuntimeError(f"{name} did not match its pinned checksum; the download was corrupted or changed.")
    os.replace(path + ".part", path)
    return path


def load_model(name):
    """A spandrel model descriptor, on the GPU in half precision when there is one."""
    import spandrel
    import torch

    model = spandrel.ModelLoader().load_from_file(fetch(name)).eval()
    model.to(state["device"])
    if state["device"] == "cuda" and model.supports_half and model.purpose != "FaceSR":
        model.half()
    return model


def open_image(path):
    """The image as uint8 RGB, plus its alpha channel (or None) and ICC profile."""
    import numpy as np
    from PIL import Image, ImageOps, UnidentifiedImageError

    if not path or not os.path.isfile(path):
        raise ValueError("Choose an image to enhance.")
    try:
        image = Image.open(path)
        image = ImageOps.exif_transpose(image)  # phone photos store their rotation in EXIF
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("That file is not an image this app can read; use PNG, JPEG or WebP.") from error
    icc = image.info.get("icc_profile")
    if image.mode in ("I", "I;16", "I;16B", "I;16L", "F"):  # 16-bit or float greyscale
        values = np.asarray(image, dtype=np.float32)
        values = values / 257.0 if values.max() > 255 else values
        image = Image.fromarray(values.clip(0, 255).astype(np.uint8))
    alpha = None
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        image = image.convert("RGBA")
        alpha = image.getchannel("A")
        if alpha.getextrema() == (255, 255):  # an alpha channel that is fully opaque
            alpha = None
    return np.asarray(image.convert("RGB")), alpha, icc


def upscale(model, image, tile, report):
    """Run the SR model tile by tile; each tile sees TILE_PAD pixels of its neighbours
    (reflected at the borders), which are cropped off, so tiles join without seams and
    memory stays bounded however large the image is."""
    import numpy as np
    import torch

    scale, (h, w) = model.scale, image.shape[:2]
    tile = tile if tile > 0 else max(h, w)
    rows, cols = math.ceil(h / tile), math.ceil(w / tile)
    tile_h, tile_w = math.ceil(h / rows), math.ceil(w / cols)  # even tiles, no thin strips
    padded = np.pad(image, ((TILE_PAD, TILE_PAD), (TILE_PAD, TILE_PAD), (0, 0)), mode="reflect")
    out = np.empty((h * scale, w * scale, 3), np.uint8)
    dtype = next(model.model.parameters()).dtype
    boxes = [(y, x) for y in range(0, h, tile_h) for x in range(0, w, tile_w)]
    for index, (y, x) in enumerate(boxes):
        report(index, len(boxes))
        y1, x1 = min(y + tile_h, h), min(x + tile_w, w)
        patch = padded[y:y1 + 2 * TILE_PAD, x:x1 + 2 * TILE_PAD]
        tensor = torch.from_numpy(np.ascontiguousarray(patch)).permute(2, 0, 1)[None]
        tensor = tensor.to(state["device"], dtype).div_(255)
        p = TILE_PAD * scale
        with torch.inference_mode():
            result = model(tensor)[0, :, p:p + (y1 - y) * scale, p:p + (x1 - x) * scale]
            result = result.float().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
        out[y * scale:y1 * scale, x * scale:x1 * scale] = result
    report(len(boxes), len(boxes))
    return out


def find_faces(image):
    """Five landmarks (eyes, nose, mouth corners) per face, largest faces first."""
    import cv2
    import numpy as np

    h, w = image.shape[:2]
    k = max(1.0, 640 / max(h, w))  # YuNet finds small faces better on an enlarged copy
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if k > 1:
        bgr = cv2.resize(bgr, (round(w * k), round(h * k)), interpolation=cv2.INTER_CUBIC)
    detector = state["yunet"]
    detector.setInputSize((bgr.shape[1], bgr.shape[0]))
    _, found = detector.detect(bgr)
    faces = []
    for row in found if found is not None else []:
        if row[2] / k < 12:  # under 12 px wide in the input: too small to restore
            continue
        points = row[4:14].reshape(5, 2) / k
        eyes, mouth = points[:2], points[3:5]
        # Order the template expects: image-left eye, image-right eye, nose, mouth corners.
        points = np.vstack([eyes[eyes[:, 0].argsort()], points[2:3], mouth[mouth[:, 0].argsort()]])
        faces.append((row[2] * row[3], points.astype(np.float32)))
    faces.sort(key=lambda face: -face[0])
    return [points for _, points in faces[:MAX_FACES]]


def restore_faces(source, result, faces, report):
    """GFPGAN v1.4 on each face, aligned from the (low-quality) source the way GFPGAN
    was trained, then blended into the upscaled result through a feathered oval."""
    import cv2
    import numpy as np
    import torch

    template = np.array(FACE_TEMPLATE, np.float32)
    mask = np.zeros((512, 512), np.float32)
    cv2.ellipse(mask, (256, 285), (178, 218), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 18)
    s = result.shape[1] / source.shape[1]
    H, W = result.shape[:2]
    canvas = result.copy()
    corners = np.array([[0, 0, 1], [511, 0, 1], [0, 511, 1], [511, 511, 1]], np.float32).T
    restored = 0
    for index, points in enumerate(faces):
        report(index, len(faces))
        matrix, _ = cv2.estimateAffinePartial2D(points, template, method=cv2.LMEDS)
        if matrix is None:
            continue
        crop = cv2.warpAffine(source, matrix, (512, 512), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))
        tensor = torch.from_numpy(crop).permute(2, 0, 1)[None].to(state["device"], torch.float32)
        with torch.inference_mode():  # GFPGAN works in [-1, 1]; spandrel's wrapper would clamp to [0, 1]
            face = state["gfpgan"].model(tensor.div(127.5).sub(1), return_rgb=False)[0][0]
            face = face.add(1).mul(127.5).clamp(0, 255).permute(1, 2, 0).float().cpu().numpy()
        # Same alignment, expressed in the result's pixels (pixel centres at +0.5).
        to_face = matrix.astype(np.float64).copy()
        to_face[:, :2] /= s
        to_face[:, 2] += matrix[:, :2] @ np.array([0.5 / s - 0.5] * 2)
        back = cv2.invertAffineTransform(to_face)
        # Paste only within the face's bounding box, not over a full-size canvas.
        box = back @ corners
        x0, y0 = np.floor(box.min(axis=1)).astype(int).clip(0, [W, H])
        x1, y1 = np.ceil(box.max(axis=1)).astype(int).clip(0, [W, H])
        if x1 <= x0 or y1 <= y0:
            continue
        back[:, 2] -= (x0, y0)
        size = (int(x1 - x0), int(y1 - y0))
        flags = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
        pasted = cv2.warpAffine(face, back, size, flags=flags)
        alpha = cv2.warpAffine(mask, back, size, flags=cv2.INTER_LINEAR)[..., None]
        region = canvas[y0:y1, x0:x1].astype(np.float32)
        canvas[y0:y1, x0:x1] = (alpha * pasted + (1 - alpha) * region).round().clip(0, 255).astype(np.uint8)
        restored += 1
    report(len(faces), len(faces))
    return canvas, restored


def comparison(before, after, limit=1024):
    """Before (the input, resized with Lanczos as any viewer would) and after, side by
    side, each at most `limit` pixels on its longest side."""
    from PIL import Image, ImageDraw, ImageFont

    w, h = after.size
    k = min(1.0, limit / max(w, h))
    size = (max(1, round(w * k)), max(1, round(h * k)))
    gap = max(4, size[0] // 100)
    frame = Image.new("RGB", (size[0] * 2 + gap, size[1]), (255, 255, 255))
    frame.paste(before.resize(size, Image.LANCZOS), (0, 0))
    frame.paste(after.resize(size, Image.LANCZOS) if k < 1 else after, (size[0] + gap, 0))
    draw = ImageDraw.Draw(frame)
    font_size = max(12, size[1] // 24)
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for text, x in (("Before", 10), ("After", size[0] + gap + 10)):
        draw.text((x, 8), text, fill=(255, 255, 255), font=font, stroke_width=max(1, font_size // 10),
                  stroke_fill=(0, 0, 0))
    return frame


started = time.time()
model_key = params.get("model", "fast")
if model_key not in MODELS:
    raise ValueError(f"Unknown model {model_key!r}; pick fast, photo or anime.")
scale = int(params.get("scale", "4"))
if scale not in (2, 4):
    raise ValueError("Scale must be 2 or 4.")
face_restore = bool(params.get("face_restore", False))
tile = max(0, int(params.get("tile", 256) or 0))
if 0 < tile < 32:
    raise ValueError("Use a tile size of at least 32 pixels, or 0 for no tiling.")
max_side = min(4096, max(64, int(params.get("max_input_px", 1024) or 1024)))
out_format = params.get("format", "png")
weights = MODELS[model_key].get(scale, MODELS[model_key][4])

state = globals().setdefault("_nzap_apps", {}).setdefault(APP, {})
models = state.setdefault("models", {})
warm = weights in models and (not face_restore or "gfpgan" in state)

if not warm:
    if "device" not in state:
        stage("install", "Installing spandrel")
        from importlib.metadata import PackageNotFoundError, version

        try:
            installed = version("spandrel")
        except PackageNotFoundError:
            installed = None
        if installed != SPANDREL:
            if "spandrel" in sys.modules:
                raise RuntimeError(f"This runtime already imported spandrel {installed}. "
                                   "Restart the runtime, then run the enhancer again.")
            pip(f"spandrel=={SPANDREL}")
        import torch

        state["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        state["device_name"] = torch.cuda.get_device_name(0) if state["device"] == "cuda" else "CPU"
    if weights not in models:
        fetch(weights)
        stage("load", f"Loading {weights.rsplit('.', 1)[0]}")
        models[weights] = load_model(weights)
    if face_restore and "gfpgan" not in state:
        import cv2

        fetch("GFPGANv1.4.pth")
        detector = fetch("face_detection_yunet_2023mar.onnx")
        stage("load", "Loading GFPGAN v1.4 and the YuNet face detector")
        state["gfpgan"] = load_model("GFPGANv1.4.pth")
        state["yunet"] = cv2.FaceDetectorYN.create(detector, "", (320, 320), 0.7, 0.3, 5000)

setup_seconds = time.time() - started
nzap("ready", "[nzap] Enhancer ready.", warm=warm, setupSeconds=round(setup_seconds, 2),
     device=state["device_name"])


def enhance():
    # A function, so the large intermediate arrays are freed when it returns instead of
    # lingering in the kernel's globals between runs.
    import numpy as np
    from PIL import Image

    run_started = time.time()
    image, alpha, icc = open_image(params.get("image"))
    in_h, in_w = image.shape[:2]
    if min(in_h, in_w) < 16:
        raise ValueError("The image is too small; it needs to be at least 16 pixels on each side.")
    if max(in_h, in_w) > max_side:  # keeps CPU runs from taking hours and RAM bounded
        k = max_side / max(in_h, in_w)
        size = (max(16, round(in_w * k)), max(16, round(in_h * k)))
        image = np.asarray(Image.fromarray(image).resize(size, Image.LANCZOS))
        alpha = alpha.resize(size, Image.LANCZOS) if alpha is not None else None
        print(f"[nzap] Scaled the {in_w}x{in_h} input down to {size[0]}x{size[1]} first "
              f"(Max input size is {max_side} px).")
    h, w = image.shape[:2]
    model = models[weights]
    share = 0.85 if face_restore else 0.97  # of the progress bar spent upscaling

    def tiles_done(done, total):
        stage("run", f"Upscaling tile {min(done + 1, total)} of {total}", progress=round(share * done / total, 3))

    result = upscale(model, image, tile, tiles_done)
    if model.scale != scale:
        result = np.asarray(Image.fromarray(result).resize((w * scale, h * scale), Image.LANCZOS))

    restored = 0
    if face_restore:
        stage("run", "Finding faces", progress=share)
        faces = find_faces(image)
        if faces:
            def faces_done(done, total):
                stage("run", f"Restoring face {min(done + 1, total)} of {total}",
                      progress=round(share + (0.97 - share) * done / total, 3))

            result, restored = restore_faces(image, result, faces, faces_done)
        else:
            print("[nzap] No faces found, so face restoration was skipped.")

    stage("run", "Saving", progress=0.97)
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(params["image"]))[0][:40] or "image"
    base = f"{OUT_DIR}/{time.strftime('%Y%m%d-%H%M%S')}-{stem}-{model_key}-x{scale}"
    enhanced = Image.fromarray(result)
    out_h, out_w = result.shape[:2]
    if alpha is not None:  # transparency is resized, not enhanced
        enhanced.putalpha(alpha.resize((out_w, out_h), Image.LANCZOS))
    extra = {"icc_profile": icc} if icc else {}
    if out_format == "jpg":
        path, mime = f"{base}.jpg", "image/jpeg"
        enhanced.convert("RGB").save(path, quality=95, subsampling=0, **extra)
    elif out_format == "webp":
        path, mime = f"{base}.webp", "image/webp"
        enhanced.save(path, quality=95, method=4, **extra)
    else:
        path, mime = f"{base}.png", "image/png"
        enhanced.save(path, compress_level=4, **extra)
    compare_path = f"{base}-compare.jpg"
    compare = comparison(Image.fromarray(image), Image.fromarray(result))
    compare.save(compare_path, quality=90)
    run_seconds = time.time() - run_started

    meta = {"width": out_w, "height": out_h, "inputWidth": in_w, "inputHeight": in_h, "scale": scale,
            "model": weights.rsplit(".", 1)[0], "faces": restored if face_restore else None,
            "seconds": round(run_seconds, 2)}
    if (w, h) != (in_w, in_h):
        meta["processedWidth"], meta["processedHeight"] = w, h
    nzap("output", f"[nzap] Saved {path} ({out_w}x{out_h})", id="image", kind="image", path=path,
         mime=mime, meta=meta)
    nzap("output", f"[nzap] Saved {compare_path}", id="compare", kind="image", path=compare_path,
         mime="image/jpeg", meta={"width": compare.width, "height": compare.height})
    faces_note = f", {restored} face{'s' if restored != 1 else ''} restored" if face_restore else ""
    nzap(
        "done",
        f"[nzap] Done in {time.time() - started:.1f}s ({w}x{h} -> {out_w}x{out_h}{faces_note}).",
        seconds={"setup": round(setup_seconds, 2), "run": round(run_seconds, 2)},
        warm=warm,
    )


enhance()
