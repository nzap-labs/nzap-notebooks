# NZAP Engine injects a `params` dict before this script runs.
# Z-Image (Tongyi-MAI) text to image from unsloth's GGUF quantizations,
# packaged as an NZAP app (see APPS.md). Tuned for a free Colab T4.
#
# The GGUF file holds only the 6B transformer; the Qwen3-4B text encoder, the
# VAE and the scheduler come from Tongyi-MAI/Z-Image. The first run on a
# runtime downloads about 14 GB and loads everything; the models stay in the
# kernel, so later runs on the same runtime only generate. Images are written
# to /content/nzap/outputs/z-image/.

import gc
import os
import random
import shutil
import subprocess
import sys
import time

from IPython.display import display

APP = "z-image"
OUT_DIR = f"/content/nzap/outputs/{APP}"
HOME = f"/content/nzap/apps/{APP}"
GGUF_REPO = "unsloth/Z-Image-GGUF"
GGUF_REVISION = "c9913e69743c5d9dfa7fdac58a0cc5709a17aa08"
BASE_REPO = "Tongyi-MAI/Z-Image"  # text encoder, tokenizer, VAE, scheduler, configs
BASE_REVISION = "04cc4abb7c5069926f75c9bfde9ef43d49423021"
DIFFUSERS = "0.40.0"  # oldest release whose ZImage GGUF loading works here
GGUF_SIZES = {"Q2_K": 3.7, "Q4_K_M": 4.7, "Q6_K": 5.7, "Q8_0": 6.7}

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


def pip(*packages):
    command = [sys.executable, "-m", "pip", "install", "-q", *packages]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"pip install {' '.join(packages)} failed:\n{result.stderr[-2000:]}")


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


# --------------------------------------------------------------------------------------
# fp16 overflow fix. Z-Image has activations far beyond the fp16 range (adaLN scales up
# to ~5e4, to_v/to_out outputs ~1e5, silu(w1 x) * (w3 x) ~1e6), so plain fp16 gives NaN
# or black images, and a T4 only has fast fp16 (bf16 is emulated). Every sub-layer is
# followed by a scale-invariant RMSNorm, so the tensors entering the risky fp16 matmuls
# can be shrunk by a scalar, and only when large. Matches an fp32 reference to 0.2%.
# --------------------------------------------------------------------------------------
def _amax_scale(x, k, per_token=False):
    """min(1, k / max|x|), per sample [B,1,1] or per token [B,S,1]; no host sync."""
    dims = (-1,) if per_token else tuple(range(1, x.ndim))
    amax = x.abs().amax(dim=dims, keepdim=True).float()
    return (k / amax.clamp_min(1e-6)).clamp(max=1.0)


def patch_fp16(transformer, k_attn=1024.0, k_out=16.0, k_ffn=64.0):
    import torch.nn as nn
    import torch.nn.functional as F

    class Rescaled(nn.Module):  # lin(x * c): safe because a scale-invariant norm follows
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x * _amax_scale(x, k_out, per_token=True).to(x.dtype))

    def forward(self, x, attn_mask, freqs_cis, adaln_input=None, noise_mask=None,
                adaln_noisy=None, adaln_clean=None):
        if noise_mask is not None:  # per-token modulation (omni/edit modes): not used here
            return self._orig_forward(x, attn_mask, freqs_cis, adaln_input, noise_mask, adaln_noisy, adaln_clean)
        n = self.attention_norm1(x)
        if self.modulation:
            mod = self.adaLN_modulation(adaln_input)
            s_msa, g_msa, s_mlp, g_mlp = mod.unsqueeze(1).chunk(4, dim=2)
            g_msa, g_mlp = g_msa.tanh(), g_mlp.tanh()
            s_msa, s_mlp = 1.0 + s_msa.float(), 1.0 + s_mlp.float()
            # attention is invariant to a per-sample input scale, so shrink by a cheap
            # upper bound on max|norm(x) * s_msa|
            bound = n.abs().amax(dim=(1, 2), keepdim=True).float() * s_msa.abs().amax(dim=(1, 2), keepdim=True)
            u = n * (s_msa * (k_attn / bound.clamp_min(1e-6)).clamp(max=1.0)).to(x.dtype)
        else:
            g_msa = g_mlp = 1.0
            u = n
        x = x + g_msa * self.attention_norm2(self.attention(u, attention_mask=attn_mask, freqs_cis=freqs_cis))
        # silu(a) * b can reach ~1e7, so each factor gets its own per-token scale
        n = self.ffn_norm1(x)
        u = n * s_mlp.to(x.dtype) if self.modulation else n
        ff = self.feed_forward
        a, b = ff.w1(u), ff.w3(u)
        r = k_ffn ** 0.5
        sa = (r / a.abs().amax(-1, keepdim=True).float().clamp_min(1e-6)).clamp(max=1.0).to(x.dtype)
        sb = (r / b.abs().amax(-1, keepdim=True).float().clamp_min(1e-6)).clamp(max=1.0).to(x.dtype)
        out = ff.w2((F.silu(a) * sa) * (b * sb))
        return x + g_mlp * self.ffn_norm2(out)

    def fast_rms_norm(self, x):  # fused kernel, ~9x faster than the multi-pass fp32 RMSNorm
        return F.rms_norm(x, x.shape[-1:], self.weight, self.eps)

    for m in transformer.modules():
        if type(m).__name__ == "RMSNorm" and getattr(m, "weight", None) is not None:
            m.forward = fast_rms_norm.__get__(m)
        if hasattr(m, "attention") and hasattr(m, "feed_forward") and not hasattr(m, "_orig_forward"):
            m._orig_forward = m.forward
            m.forward = forward.__get__(m)
            m.attention.to_out[0] = Rescaled(m.attention.to_out[0])


class WeightCache:
    """diffusers dequantizes every GGUF weight on every forward (~40% of a step on a T4).
    `fill()` keeps as many blocks as fit as plain fp16 in VRAM, leaving `reserve` GiB free
    for activations and the VAE decode, and `drop()` hands the VRAM back while the text
    encoder runs. The quantized weights of cached blocks are saved to disk once and come
    back through the page cache, which (unlike process memory) never gets the kernel
    killed for running out of RAM."""

    def __init__(self, transformer, store):
        self.blocks = [m for m in transformer.modules() if hasattr(m, "attention") and hasattr(m, "feed_forward")]
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
    """Qwen3-4B, 4-bit (NF4) so it can wait on the CPU between runs and visit the GPU
    only to encode the prompt."""
    import torch
    from diffusers import ZImagePipeline
    from transformers import AutoModel, BitsAndBytesConfig

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    te = AutoModel.from_pretrained(BASE_REPO, revision=BASE_REVISION, subfolder="text_encoder",
                                   dtype=torch.float16, device_map="cuda", quantization_config=quant)
    pipe = ZImagePipeline.from_pretrained(BASE_REPO, revision=BASE_REVISION, transformer=None, vae=None,
                                          text_encoder=te, dtype=torch.float16)
    return pipe


def load_pipeline(quant):
    import torch
    from diffusers import GGUFQuantizationConfig, ZImagePipeline, ZImageTransformer2DModel
    from huggingface_hub import hf_hub_download

    stage("download", f"Downloading the {quant} transformer ({GGUF_SIZES[quant]} GB)")
    path = hf_hub_download(GGUF_REPO, f"z-image-{quant}.gguf", revision=GGUF_REVISION, token=hf_token)
    stage("load", f"Loading Z-Image {quant} onto the GPU")
    transformer = ZImageTransformer2DModel.from_single_file(
        path, quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16), dtype=torch.float16,
        config=BASE_REPO, subfolder="transformer", revision=BASE_REVISION)
    patch_fp16(transformer)
    pipe = ZImagePipeline.from_pretrained(BASE_REPO, revision=BASE_REVISION, transformer=transformer,
                                          text_encoder=None, tokenizer=None, dtype=torch.float16)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe, WeightCache(transformer, f"{HOME}/blocks-{quant}")


started = time.time()
prompt = (params.get("prompt") or "").strip()
if not prompt:
    raise ValueError("Describe the image you want.")
negative = (params.get("negative_prompt") or "").strip()
width, height = (int(v) for v in params.get("size", "768x768").split("x"))
quant = params.get("quality", "Q6_K")
steps = max(1, int(params.get("steps", 28)))
cfg = float(params.get("guidance", 4.0))
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
        raise RuntimeError("Z-Image needs a GPU runtime (T4 or better). Switch the runtime to a GPU.")
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
        stage("install", "Installing diffusers and GGUF support")
        from importlib.metadata import version

        from packaging.version import Version

        # Colab ships diffusers 0.40; a newer one (e.g. from the Qwen-Image app) works too.
        if Version(version("diffusers")) < Version(DIFFUSERS):
            pip(f"diffusers=={DIFFUSERS}")
        pip("gguf==0.19.0", "bitsandbytes==0.50.2")
        loaded = sys.modules.get("diffusers")
        if loaded is not None and not hasattr(loaded, "ZImagePipeline"):
            raise RuntimeError(
                f"This runtime already imported diffusers {loaded.__version__}. "
                "Restart the runtime, then run Z-Image again."
            )
        stage("download", "Downloading the Qwen3 text encoder and VAE (8 GB)")
        from huggingface_hub import snapshot_download

        snapshot_download(BASE_REPO, revision=BASE_REVISION, token=hf_token, allow_patterns=[
            "model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/*", "vae/*", "transformer/config.json"])
    else:
        # Loading a GGUF file peaks near 9 GB of the T4 VM's 12.7 GB RAM, so the text
        # encoder waits on the GPU meanwhile.
        state["encoder"].text_encoder.to("cuda")

    state["pipe"], state["cache"] = load_pipeline(quant)
    release_memory()  # hand back the GGUF loading buffers before the text encoder returns to RAM
    if "encoder" not in state:
        stage("load", "Loading the text encoder")
        state["encoder"] = load_text_encoder()
    state["encoder"].text_encoder.to("cpu")
    release_memory()
    state["quant"] = quant
    state["gpu"] = torch.cuda.get_device_name(0)

setup_seconds = time.time() - started
nzap("ready", f"[nzap] Z-Image {quant} ready.", warm=warm, setupSeconds=round(setup_seconds, 2), device=state["gpu"])


def generate():
    # A function, so no pipeline reference lingers in the kernel's globals after the run
    # (it would keep the old model in VRAM when the next run switches quantization).
    import torch

    run_started = time.time()
    torch.cuda.reset_peak_memory_stats()
    pipe, cache, encoder = state["pipe"], state["cache"], state["encoder"]

    # The text encoder borrows the VRAM of the fp16 weight cache, then hands it back.
    stage("run", "Encoding the prompt", progress=0.0)
    cache.drop()
    encoder.text_encoder.to("cuda")
    with torch.no_grad():
        pe, ne = encoder.encode_prompt(prompt, device="cuda", do_classifier_free_guidance=cfg > 1,
                                       negative_prompt=negative)
    encoder.text_encoder.to("cpu")
    release_memory()
    # VRAM kept free for activations and the VAE decode; the rest caches fp16 weights.
    reserve = max(3.0, 1.5 + 3.0 * width * height / 2**20)
    cached, total = cache.fill(reserve)
    print(f"[nzap] {cached}/{total} transformer blocks cached as fp16 in VRAM")

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    label = "Generating"
    for index in range(count):
        image_seed = seed + index
        if count > 1:
            label = f"Generating image {index + 1} of {count}"

        def on_step(_pipe, step, _timestep, kwargs, index=index):
            stage("run", label, progress=round((index * steps + step + 1) / (steps * count), 3))
            return kwargs

        image = pipe(
            prompt_embeds=[e.to("cuda") for e in pe],
            negative_prompt_embeds=[e.to("cuda") for e in ne],
            width=width, height=height, num_inference_steps=steps, guidance_scale=cfg,
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
