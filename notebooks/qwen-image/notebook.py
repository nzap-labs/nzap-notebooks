# NZAP Engine injects a `params` dict before this script runs.
# Qwen-Image-2.1 text to image from unsloth's GGUF quantizations, packaged as
# an NZAP app (see APPS.md). Tuned for a free Colab T4.
#
# The GGUF file holds only the transformer; the Qwen3-VL-8B text encoder
# (loaded 4-bit), the VAE and the scheduler come from Qwen/Qwen-Image-2.1. The
# first run on a runtime installs a pinned diffusers commit, downloads about
# 22 GB and loads everything; the models stay in the kernel, so later runs on
# the same runtime only generate. Images are written to
# /content/nzap/outputs/qwen-image/.

import gc
import json
import os
import random
import shutil
import subprocess
import sys
import time

from IPython.display import display

APP = "qwen-image"
OUT_DIR = f"/content/nzap/outputs/{APP}"
HOME = f"/content/nzap/apps/{APP}"
GGUF_REPO = "unsloth/Qwen-Image-2.1-GGUF"
GGUF_REVISION = "2c31ccd392b367a6637841a143813320a02dff55"
BASE_REPO = "Qwen/Qwen-Image-2.1"  # text encoder, processor, VAE, scheduler, configs
BASE_REVISION = "d26bb61231c349cf6b7896fa83353113880e1ba3"
# QwenImage21Pipeline is not in a diffusers release yet: pinned to a reviewed commit of main.
DIFFUSERS_COMMIT = "8b33bfc04b6b5e8bb58a58e55f68746c1bbee4cd"
GGUF_SIZES = {"Q4_K_M": 3.9, "Q6_K": 5.8}

# Colab's huggingface_hub otherwise waits on the notebook-UI secret store.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
# Moving the text encoder in and out of VRAM fragments the default allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def nzap(event, text, **fields):
    """One NZAP app event; `text/plain` is what the console shows."""
    payload = {"v": 1, "app": APP, "event": event, **fields}
    display({"application/vnd.nzap.app+json": payload, "text/plain": text}, raw=True)


def stage(stage_id, label, progress=None):
    nzap("stage", f"[nzap] {label}…", id=stage_id, label=label, progress=progress)


def pip(*args):
    command = [sys.executable, "-m", "pip", *args]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"pip {' '.join(args)} failed:\n{result.stderr[-2000:]}")


def release_memory():
    """Return freed VRAM and heap to the system (the T4 VM has only 12.7 GB of RAM)."""
    import ctypes

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def installed_diffusers_commit():
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        origin = json.loads(distribution("diffusers").read_text("direct_url.json") or "{}")
    except PackageNotFoundError:
        return None
    return origin.get("vcs_info", {}).get("commit_id")


def install():
    from importlib.metadata import version

    from packaging.version import Version

    if installed_diffusers_commit() != DIFFUSERS_COMMIT:
        pip("uninstall", "-y", "-q", "torchao")  # Colab's torchao is too old for diffusers main
        pip("install", "-q", f"git+https://github.com/huggingface/diffusers@{DIFFUSERS_COMMIT}")
    if Version(version("transformers")) < Version("5.17"):
        pip("install", "-q", "transformers==5.17.0")
    pip("install", "-q", "gguf==0.19.0", "bitsandbytes==0.50.2")
    loaded = sys.modules.get("diffusers")
    if loaded is not None and not hasattr(loaded, "QwenImage21Pipeline"):
        raise RuntimeError(
            f"This runtime already imported diffusers {loaded.__version__}, which predates Qwen-Image-2.1. "
            "Restart the runtime, then run Qwen-Image again."
        )
    loaded = sys.modules.get("transformers")
    if loaded is not None and Version(loaded.__version__) < Version("5.17"):
        raise RuntimeError(
            f"This runtime already imported transformers {loaded.__version__}. "
            "Restart the runtime, then run Qwen-Image again."
        )


def dequant_non_linear(model):
    """Some tensors (e.g. txt_in.text_norm.weight) are stored as BF16 in the GGUF, which diffusers
    exposes as raw uint8 bytes. Only GGUFLinear knows how to dequantize, so fix the rest here."""
    import torch
    import torch.nn as nn
    from diffusers.quantizers.gguf.utils import GGUFLinear, dequantize_gguf_tensor

    for m in model.modules():
        if isinstance(m, GGUFLinear):
            continue
        for name, p in list(m.named_parameters(recurse=False)):
            if hasattr(p, "quant_type"):
                setattr(m, name, nn.Parameter(dequantize_gguf_tensor(p).to(torch.float16), requires_grad=False))


class WeightCache:
    """diffusers dequantizes every GGUF weight on every forward (~20% of a step on a T4).
    `fill()` keeps as many transformer blocks as fit as plain fp16 in VRAM, leaving
    `reserve` GiB free for activations, and `drop()` hands the VRAM back to the text
    encoder and to the untiled VAE decode (~7 GiB per megapixel). The quantized weights of
    cached blocks are saved to disk once and come back through the page cache, which
    (unlike process memory) never gets the kernel killed for running out of RAM."""

    def __init__(self, transformer, store):
        self.blocks = list(transformer.transformer_blocks)
        self.store = store
        self.cached = []  # (block index, [(GGUFLinear, quant type)])
        os.makedirs(store, exist_ok=True)

    def fill(self, reserve_gib):
        import torch
        import torch.nn as nn
        from diffusers.quantizers.gguf.utils import GGUFLinear, dequantize_gguf_tensor

        for index in range(len(self.cached), len(self.blocks)):
            linears = [m for m in self.blocks[index].modules() if isinstance(m, GGUFLinear)]
            need = sum(m.in_features * m.out_features * 2 for m in linears)
            torch.cuda.empty_cache()
            if torch.cuda.mem_get_info()[0] - need < reserve_gib * 2**30:
                break
            path = f"{self.store}/{index}.pt"
            if not os.path.exists(path):
                torch.save([m.weight.detach().cpu().as_tensor() for m in linears], path + ".tmp")
                os.replace(path + ".tmp", path)
            entry = []
            for m in linears:
                entry.append((m, m.weight.quant_type))
                m.weight = nn.Parameter(dequantize_gguf_tensor(m.weight).to(torch.float16), requires_grad=False)
                m.forward = nn.Linear.forward.__get__(m)
            self.cached.append((index, entry))
        torch.cuda.empty_cache()
        return len(self.cached), len(self.blocks)

    def drop(self):
        import torch
        from diffusers.quantizers.gguf.utils import GGUFParameter

        for index, entry in self.cached:
            saved = torch.load(f"{self.store}/{index}.pt", mmap=True, weights_only=True)
            for (m, quant_type), data in zip(entry, saved):
                m.weight = GGUFParameter(data.to("cuda"), quant_type=quant_type)
                m.__dict__.pop("forward", None)
        self.cached.clear()
        torch.cuda.empty_cache()


def load_text_encoder():
    """Qwen3-VL-8B in 4-bit (NF4): it waits on the CPU between runs and visits the GPU
    only to encode the prompt."""
    import torch
    from diffusers import QwenImage21Pipeline
    from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    te = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_REPO, revision=BASE_REVISION, subfolder="text_encoder", dtype=torch.float16,
        device_map="cuda", quantization_config=quant)
    return QwenImage21Pipeline.from_pretrained(BASE_REPO, revision=BASE_REVISION, transformer=None, vae=None,
                                               text_encoder=te, dtype=torch.float16)


def load_pipeline(quant):
    import torch
    from diffusers import GGUFQuantizationConfig, QwenImage21Pipeline, QwenImage21Transformer2DModel
    from huggingface_hub import hf_hub_download

    stage("download", f"Downloading the {quant} transformer ({GGUF_SIZES[quant]} GB)")
    path = hf_hub_download(GGUF_REPO, f"qwen-image-2.1-{quant}.gguf", revision=GGUF_REVISION, token=hf_token)
    stage("load", f"Loading Qwen-Image {quant} onto the GPU")
    transformer = QwenImage21Transformer2DModel.from_single_file(
        path, quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16), dtype=torch.float16,
        config=BASE_REPO, subfolder="transformer", revision=BASE_REVISION)
    dequant_non_linear(transformer)
    pipe = QwenImage21Pipeline.from_pretrained(BASE_REPO, revision=BASE_REVISION, transformer=transformer,
                                               text_encoder=None, dtype=torch.float16)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    cache = WeightCache(transformer, f"{HOME}/blocks-{quant}")
    decode = pipe.vae.decode

    def decode_without_cache(*args, **kwargs):  # the untiled VAE decode needs the VRAM back
        cache.drop()
        return decode(*args, **kwargs)

    pipe.vae.decode = decode_without_cache
    return pipe, cache


started = time.time()
prompt = (params.get("prompt") or "").strip()
if not prompt:
    raise ValueError("Describe the image you want.")
negative = (params.get("negative_prompt") or "").strip() or " "
width, height = (int(v) for v in params.get("size", "1024x1024").split("x"))
quant = params.get("quality", "Q4_K_M")
steps = max(1, int(params.get("steps", 20)))
cfg = float(params.get("guidance", 6.0))
use_cfg = cfg > 1
count = max(1, min(4, int(params.get("num_images", 1))))
seed = int(params.get("seed", -1))
if seed < 0:
    seed = random.randint(0, 2**31 - 1)
# Optional: Hugging Face's authenticated rate limits make the first download much faster.
# The token is only passed to the Hugging Face downloads below.
hf_token = (params.get("hf_token") or "").strip() or None

state = globals().setdefault("_nzap_apps", {}).setdefault(APP, {})
warm = state.get("quant") == quant

if not warm:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen-Image needs a GPU runtime (T4 or better). Switch the runtime to a GPU.")
    if "pipe" in state:  # another quantization was loaded: free it first
        state["cache"].drop()
        shutil.rmtree(state["cache"].store, ignore_errors=True)
        del state["pipe"], state["cache"]
        state.pop("quant", None)
        for name in ("last_exc", "last_value", "last_traceback"):  # a failed run's frames pin it too
            if hasattr(sys, name):
                setattr(sys, name, None)
        release_memory()

    if "encoder" not in state:
        stage("install", "Installing diffusers (pinned main) and GGUF support")
        install()
        stage("download", "Downloading the Qwen3-VL text encoder and VAE (17 GB)")
        from huggingface_hub import snapshot_download

        snapshot_download(BASE_REPO, revision=BASE_REVISION, token=hf_token, allow_patterns=[
            "model_index.json", "scheduler/*", "processor/*", "text_encoder/*", "vae/*", "transformer/config.json"])
    else:
        # Loading a GGUF file takes several GB of the T4 VM's 12.7 GB RAM, so the text
        # encoder waits on the GPU meanwhile.
        state["encoder"].text_encoder.to("cuda")

    state["pipe"], state["cache"] = load_pipeline(quant)
    release_memory()  # hand back the GGUF loading buffers before the text encoder returns to RAM
    if "encoder" not in state:
        stage("load", "Loading the text encoder (4-bit)")
        state["encoder"] = load_text_encoder()
    state["encoder"].text_encoder.to("cpu")
    release_memory()
    state["quant"] = quant
    state["gpu"] = torch.cuda.get_device_name(0)

setup_seconds = time.time() - started
nzap("ready", f"[nzap] Qwen-Image {quant} ready.", warm=warm, setupSeconds=round(setup_seconds, 2),
     device=state["gpu"])


def generate():
    # A function, so no pipeline reference lingers in the kernel's globals after the run
    # (it would keep the old model in VRAM when the next run switches quantization).
    import torch

    run_started = time.time()
    torch.cuda.reset_peak_memory_stats()
    pipe, cache, encoder = state["pipe"], state["cache"], state["encoder"]

    stage("run", "Encoding the prompt", progress=0.0)
    cache.drop()
    encoder.text_encoder.to("cuda")
    with torch.no_grad():
        pe, pm, _ = encoder.encode_prompt(prompt, device="cuda")
        ne, nm, _ = encoder.encode_prompt(negative, device="cuda") if use_cfg else (None, None, None)
    encoder.text_encoder.to("cpu")
    release_memory()


    def dev(t):
        return None if t is None else t.to("cuda")


    # VRAM kept free for activations while denoising; the cache is dropped before each decode.
    reserve = 2.5 + 1.5 * width * height / 2**20
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    label = "Generating"
    for index in range(count):
        image_seed = seed + index
        if count > 1:
            label = f"Generating image {index + 1} of {count}"
        cached, total = cache.fill(reserve)
        print(f"[nzap] {cached}/{total} transformer blocks cached as fp16 in VRAM")

        def on_step(_pipe, step, _timestep, kwargs, index=index):
            stage("run", label, progress=round((index * steps + step + 1) / (steps * count), 3))
            return kwargs

        image = pipe(
            prompt_embeds=dev(pe), prompt_embeds_mask=dev(pm),
            negative_prompt_embeds=dev(ne), negative_prompt_embeds_mask=dev(nm),
            true_cfg_scale=cfg, width=width, height=height, num_inference_steps=steps,
            generator=torch.Generator("cuda").manual_seed(image_seed),
            callback_on_step_end=on_step,
        ).images[0]
        path = f"{OUT_DIR}/{stamp}-{image_seed}.png"
        image.save(path)
        nzap(
            "output",
            f"[nzap] Saved {path}",
            id="image",
            kind="image",
            path=path,
            mime="image/png",
            meta={"width": width, "height": height, "seed": image_seed, "steps": steps,
                  "guidance": cfg, "quality": quant, "prompt": prompt},
        )

    run_seconds = time.time() - run_started
    nzap(
        "done",
        f"[nzap] Done in {time.time() - started:.1f}s ({run_seconds / count:.1f}s per image, "
        f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB peak VRAM).",
        seconds={"setup": round(setup_seconds, 2), "run": round(run_seconds, 2)},
        warm=warm,
    )


generate()
