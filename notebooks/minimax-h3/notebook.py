# NZAP Engine injects a `params` dict before this script runs.
# MiniMax-H3 text to video with matching stereo sound on a Colab TPU, as an NZAP app (see APPS.md).
#
# The JAX/Pallas pipeline (GGUF dequantization, DiT, text encoder, video/audio VAEs) is embedded verbatim
# below from https://github.com/Rahulsingh1939/inference_h3 (written by h3/make_nzap_notebook.py). It runs
# as a subprocess because importing jax in this kernel would claim the TPU and the pipeline could then
# not open it ("/dev/vfio/0: Device or resource busy"). Its log lines become app events.

import glob
import os
import re
import subprocess
import sys
import time

from IPython.display import display

APP = "minimax-h3"
ROOT = "/content/nzap/minimax-h3"
OUT_DIR = "/content/nzap/outputs/minimax-h3"
RAM_DIR = "/dev/shm/h3w"
JAX_VERSION, LIBTPU_VERSION, GGUF_VERSION = "0.7.2", "0.0.23", "0.19.0"
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

SOURCES = {
    'inference_minimax_h3.py': r'''#!/usr/bin/env python
"""MiniMax-H3 text -> video + stereo audio on a TPU (JAX/Pallas), from unsloth/MiniMax-H3-GGUF.

    python inference_minimax_h3.py --prompt "a red panda stepping along a mossy log in a misty forest"
    python inference_minimax_h3.py --prompt "..." --quant Q4_K --turbo 4 --width 672 --height 384
    python inference_minimax_h3.py --prompt "..." --turbo none --steps 30        # base model, no LoRA

Stages (each frees its device memory before the next one loads):
  1. Qwen3-VL-32B conditioner (GGUF, 50 layers)       -> prompt embeddings
  2. pruned DiT (GGUF, weights stay quantized on TPU)  -> video + audio latents (rectified flow)
  3. ViT video decoder + BigVGAN audio decoder         -> frames + 32 kHz stereo -> .mp4
"""
import argparse
import gc
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "h3"))

GGUF_REPO, BASE_REPO = "unsloth/MiniMax-H3-GGUF", "MiniMaxAI/MiniMax-H3"
# pinned Hugging Face commits (weights, tokenizer, turbo LoRAs) so a rerun downloads exactly what was tested
REVISION = {GGUF_REPO: "d629413c2e5b51b38c453668b75ca3b06ca92703",
            BASE_REPO: "42ed227ee7df40d41602854ae760620d6eb651fe",
            "Comfy-Org/MiniMax-H3": "e5eb578a89295337b8ff433a035929ce0279e0b6"}
DIT_QUANTS = ["Q2_K", "UD-Q2_K_XL", "Q3_K", "UD-Q3_K_XL", "Q4_K", "Q5_0", "Q6_K", "Q8_0"]
LORA_REPO = "Comfy-Org/MiniMax-H3"
# Comfy-Org's distilled turbo LoRAs (lightx2v v1.0): file, steps, video/audio shift they were distilled with
TURBO = {"8": ("loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", 8, 12.0, 3.0),
         "4": ("loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors", 4, 6.0, 3.0)}


def parse_args():
    p = argparse.ArgumentParser(description="MiniMax-H3 GGUF text-to-video+audio on TPU")
    p.add_argument("--prompt", required=True)
    p.add_argument("--quant", default="Q2_K", choices=DIT_QUANTS, help="DiT GGUF quantization")
    p.add_argument("--te_quant", default="Q2_K_M", choices=["Q2_K_M", "Q4_K_M"], help="text encoder GGUF")
    p.add_argument("--width", type=int, default=672)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--frames", type=int, default=124, help="24 fps; snapped up to 17n+5 (124 = 5.2 s)")
    p.add_argument("--turbo", default="8", choices=["8", "4", "none"],
                   help="distilled turbo LoRA: 8 steps (544p-trained) or 4 steps (768p-trained); none = base model")
    p.add_argument("--lora", default=None, help="extra/other ComfyUI-format DiT LoRA (local path or repo file)")
    p.add_argument("--lora_strength", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=None, help="denoiser evaluations (default: turbo preset, else 30)")
    p.add_argument("--flow_shift", type=float, default=None, help="video shift (default: turbo preset, else 12)")
    p.add_argument("--audio_flow_shift", type=float, default=None, help="audio shift (default: preset, else 3)")
    p.add_argument("--audio_device", default="cpu", choices=["cpu", "tpu"],
                   help="BigVGAN decode device: the TPU VM's host CPU compiles the fp32 conv stack far faster")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="outputs/minimax_h3.mp4")
    p.add_argument("--attn", default="auto", choices=["auto", "pallas", "chunked", "xla"],
                   help="pallas = TPU flash attention; chunked = plain-XLA query-blocked fallback")
    p.add_argument("--weights_dir", default=os.environ.get("H3_WEIGHTS", "h3_weights"),
                   help="where the GGUF/VAE files are downloaded to (reused if present)")
    p.add_argument("--ram_dir", default="auto",
                   help="download the two GGUFs (~20 GB at Q2) into this tmpfs dir and memory-map them: Colab TPU "
                        "VMs download from HF at ~120 MB/s but read their disk at ~27 MB/s. auto = /dev/shm/h3w "
                        "when it has room, 'none' = keep everything in --weights_dir")
    p.add_argument("--cache_dir", default="h3_cache",
                   help="prompt embeddings / final latents are cached here so reruns skip finished stages")
    p.add_argument("--debug_layers", type=int, default=0, help="truncate every stack (plumbing tests only)")
    a = p.parse_args()
    lora, steps, shift, ashift = TURBO.get(a.turbo, (None, 30, 12.0, 3.0))
    a.lora = a.lora or lora
    a.steps = a.steps or steps
    a.flow_shift = a.flow_shift if a.flow_shift is not None else shift
    a.audio_flow_shift = a.audio_flow_shift if a.audio_flow_shift is not None else ashift
    return a


def log(msg, t0=[time.time()]):  # noqa: B006
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


def free(tree):
    import jax
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "delete"):
            leaf.delete()
    gc.collect()


def ram_dir(args, names):
    """The tmpfs directory to hold the GGUFs, or None when there is none / it is too small."""
    import shutil
    if args.ram_dir == "none":
        return None
    d = "/dev/shm/h3w" if args.ram_dir == "auto" else args.ram_dir
    if not os.path.isdir(os.path.dirname(d.rstrip("/")) or "/"):
        return None
    from huggingface_hub import HfApi
    need = sum(i.size for i in HfApi().get_paths_info(GGUF_REPO, names, revision=REVISION[GGUF_REPO])
               if not os.path.exists(os.path.join(d, i.path)))
    free = shutil.disk_usage(os.path.dirname(d.rstrip("/"))).free
    if need > free - (1 << 30):
        log(f"{d}: {free / 2**30:.1f} GiB free < {need / 2**30:.1f} GiB needed; GGUFs stay in {args.weights_dir}")
        return None
    return d


def fetch(args):
    from huggingface_hub import hf_hub_download
    dl = lambda f, d=args.weights_dir: hf_hub_download(GGUF_REPO, f, revision=REVISION[GGUF_REPO], local_dir=d)  # noqa: E731
    ggufs = [f"minimax_h3_fl2va_pruned-{args.quant}.gguf", f"qwen3vl_32b_minimax_h3-{args.te_quant}.gguf"]
    gd = ram_dir(args, ggufs) or args.weights_dir
    t0 = time.time()
    paths = dict(dit=dl(ggufs[0], gd), te=dl(ggufs[1], gd),
                 vae=dl("vae/minimax_h3_video_vae_fp16.safetensors"),
                 avae=dl("vae/minimax_h3_audio_vae_fp32.safetensors"))
    if args.lora:
        paths["lora"] = args.lora if os.path.exists(args.lora) else hf_hub_download(
            LORA_REPO, args.lora, revision=REVISION[LORA_REPO], local_dir=args.weights_dir)
    log(f"weights ready in {time.time() - t0:.0f}s (GGUFs in {gd})")
    return paths


def warm_page_cache(files):
    """Read the VAE files once in a background thread while the TE/DiT stages run, so the decode stage
    finds them in the page cache instead of on the VM's slow disk (5.2 GB ViT VAE ~ 3 min at 25 MiB/s)."""
    import threading

    def run():
        for f in files:
            with open(f, "rb", buffering=0) as fh:
                while fh.read(64 << 20):
                    pass
    threading.Thread(target=run, daemon=True).start()


def ensure_deps():
    """Install the light pure-python deps with *this* interpreter (pip/python can differ in a terminal)."""
    import importlib.util
    missing = [m for m in ("gguf", "safetensors", "transformers", "ml_dtypes") if importlib.util.find_spec(m) is None]
    if missing:
        print(f"installing {missing} into {sys.executable}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


def pallas_works():
    """Compile + run a tiny Pallas flash attention; a jax/libtpu version mismatch fails here."""
    import jax.numpy as jnp
    from attention_tpu import flash_attention_padded
    try:
        x = jnp.ones((700, 2, 128), jnp.bfloat16)
        flash_attention_padded(x, x, x).block_until_ready()
        return True
    except Exception as e:  # noqa: BLE001
        log(f"Pallas flash attention unavailable ({str(e).splitlines()[0][:120]}); using chunked XLA attention")
        return False


def main():
    args = parse_args()
    ensure_deps()
    import numpy as np
    import jax
    import jax.numpy as jnp
    from transformers import AutoTokenizer
    from gguf_jax import GGUFFile
    import te_jax as TE
    import dit_jax as D

    if args.attn != "auto":
        impl = args.attn
    elif jax.default_backend() == "tpu":
        impl = "pallas" if pallas_works() else "chunked"
    else:
        impl = "xla"
    nl = args.debug_layers or None
    log(f"backend {jax.default_backend()} {jax.devices()[0].device_kind}, attention={impl}, "
        f"lora={args.lora and os.path.basename(args.lora)} x{args.lora_strength}, steps={args.steps}, "
        f"shift={args.flow_shift}/{args.audio_flow_shift}")
    paths = fetch(args)
    warm_page_cache([paths["vae"], paths["avae"]])

    import hashlib
    os.makedirs(args.cache_dir, exist_ok=True)
    tag = lambda *a: hashlib.sha1(repr(a).encode()).hexdigest()[:16]  # noqa: E731
    te_cache = os.path.join(args.cache_dir, f"te_{tag(args.prompt, args.te_quant, nl)}.npy")
    lat_cache = os.path.join(args.cache_dir, f"lat_{tag(args.prompt, args.te_quant, nl, args.quant, args.width, args.height, args.frames, args.steps, args.flow_shift, args.audio_flow_shift, args.seed, args.lora, args.lora_strength)}.npz")

    # ------------------------------------------------------------------ 1. conditioner
    tok = AutoTokenizer.from_pretrained(BASE_REPO, subfolder="tokenizer", revision=REVISION[BASE_REPO])
    ids = tok(args.prompt, add_special_tokens=False)["input_ids"]
    if os.path.exists(te_cache):
        prompt_embeds = np.load(te_cache)
        log(f"[1/3] prompt embeddings from cache ({len(ids)} tokens)")
    else:
        te_file = GGUFFile(paths["te"]).preload(log)
        x = TE.embed(te_file, ids)
        layers = TE.load_te_layers(te_file, jax.device_put, None if nl is None else range(nl), log=log)
        log(f"[1/3] text encoder loaded ({len(ids)} tokens)")
        cos, sin = TE.rope_cos_sin(len(ids))
        prompt_embeds = np.asarray(TE.te_forward(layers, jnp.asarray(x), cos, sin).astype(jnp.float32))
        free(layers); te_file.buf = None; del te_file
        np.save(te_cache, prompt_embeds)
        log("[1/3] prompt encoded")

    # ------------------------------------------------------------------ 2. denoise
    frames = D.align_num_frames(args.frames)
    T, lh, lw = D.latent_frames(frames), args.height // 16, args.width // 16
    A = int(round(frames / D.FPS * D.AUDIO_LATENTS_PER_SECOND))
    assert args.height % 32 == 0 and args.width % 32 == 0, "height/width must be multiples of 32"
    if os.path.exists(lat_cache):
        c = np.load(lat_cache)
        video, audio = c["video"], c["audio"]
        log("[2/3] latents from cache")
    else:
        video, audio = denoise(args, paths, prompt_embeds, len(ids), T, lh, lw, A, nl, impl)
        np.savez(lat_cache, video=video, audio=audio)
    assert np.isfinite(video).all() and np.isfinite(audio).all(), "non-finite latents"
    decode(args, paths, video, audio, T, lh, lw, A, nl)


def denoise(args, paths, prompt_embeds, n_text, T, lh, lw, A, nl, impl):
    import numpy as np
    import jax
    import jax.numpy as jnp
    from gguf_jax import GGUFFile
    import dit_jax as D
    dit_file = GGUFFile(paths["dit"]).preload(log)
    params = D.load_dit(dit_file, jax.device_put, nl, log=log)
    dit_file.buf = None
    if args.lora:
        D.add_lora(params, paths["lora"], args.lora_strength, jax.device_put, log=log)
    log(f"[2/3] DiT {args.quant} loaded")
    lay = D.build_layout(n_text, T, lh, lw, A)
    cos, sin = D.rope_cos_sin(lay["position_ids"], params["inv_freq"])
    jl = {k: (jnp.asarray(v) if isinstance(v, np.ndarray) and k != "position_ids" else v) for k, v in lay.items()}
    text = jax.jit(lambda p, e: D.encode_text(p, e, impl))(params, jnp.asarray(prompt_embeds))

    key = jax.random.PRNGKey(args.seed)
    kv, ka = jax.random.split(key)
    video = D.patchify(jax.random.normal(kv, (24, T, lh, lw), jnp.float32))
    audio = jax.random.normal(ka, (2 * A, 32), jnp.float32)
    sv, tv = D.schedule(args.steps + 1, args.flow_shift)
    sa, ta = D.schedule(args.steps + 1, args.audio_flow_shift)

    @jax.jit
    def step(params, text, video, audio, t_v, t_a, s_v, s_vn, s_a, s_an):
        vv, va = D.forward(params, text, video, audio, t_v, t_a, jl, cos, sin, impl)
        return D.euler_step(video, vv, t_v, s_v, s_vn), D.euler_step(audio, va, t_a, s_a, s_an)

    for i in range(len(tv)):
        t1 = time.time()
        video, audio = step(params, text, video, audio, tv[i], ta[i], sv[i], sv[i + 1], sa[i], sa[i + 1])
        video.block_until_ready()
        log(f"[2/3] step {i + 1}/{len(tv)} t={tv[i]:.4f} ({time.time() - t1:.1f}s)")
    video, audio = np.asarray(video), np.asarray(audio)
    free(params); del dit_file
    return video, audio


def decode(args, paths, video, audio, T, lh, lw, A, nl):
    import numpy as np
    import jax
    import jax.numpy as jnp
    import dit_jax as D
    import vae_jax as V
    # ------------------------------------------------------------------ 3. decode
    vp, (lmean, lstd) = V.load_video_vae(paths["vae"], jax.device_put, nl)
    z = np.asarray(D.unpatchify(jnp.asarray(video), 24, T, lh, lw))[None]
    z = z * lstd[None, :, None, None, None] + lmean[None, :, None, None, None]
    dec = jax.jit(lambda p, zt: V.vit_decode(p, V.post_quant_conv(p, zt)))
    px = V.decode_video(z, lambda zt: np.asarray(dec(vp, jnp.asarray(zt))))
    free(vp)
    mean = np.array([0.485, 0.456, 0.406], np.float32)[None, :, None, None, None]
    std = np.array([0.229, 0.224, 0.225], np.float32)[None, :, None, None, None]
    frames_u8 = (np.clip(px * std + mean, 0, 1)[0].transpose(1, 2, 3, 0) * 255).round().astype(np.uint8)
    log(f"[3/3] video decoded {frames_u8.shape}")

    dev = jax.devices("cpu")[0] if args.audio_device == "cpu" else jax.devices()[0]
    t0 = time.time()
    with jax.default_device(dev):
        ap, (amean, astd) = V.load_audio_vae(paths["avae"], lambda a: jax.device_put(a, dev))
        alat = audio.reshape(2, A, 32).transpose(0, 2, 1) * astd[None, :, None] + amean[None, :, None]
        wav = np.asarray(V.audio_decode(ap, jax.device_put(alat, dev)))[:, 0]   # [2, samples]
    free(ap)
    log(f"[3/3] audio decode on {dev.platform}: {time.time() - t0:.1f}s")
    log(f"[3/3] audio decoded {wav.shape}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    wav_path = args.output[:-4] + ".wav"
    pcm = (np.clip(wav.T, -1, 1) * 32767).astype("<i2")
    import wave
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(32000); w.writeframes(pcm.tobytes())
    n, h, w_ = frames_u8.shape[:3]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w_}x{h}",
           "-r", "24", "-i", "-", "-i", wav_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-c:a", "aac", "-b:a", "192k", "-shortest", args.output]
    subprocess.run(cmd, input=frames_u8.tobytes(), check=True)
    log(f"saved {args.output} ({n} frames {w_}x{h} @24fps + 32 kHz stereo)")


if __name__ == "__main__":
    main()
''',
    'h3/gguf_jax.py': r'''"""GGUF loading + dequantization for JAX/TPU.

Quantized weights stay in their raw GGML block layout on the device (uint8, `[out, n_blocks, block_bytes]`)
and are expanded to bf16 inside the jitted forward, right before each matmul, so HBM only ever holds
the quantized bytes plus one layer's dequantized weight at a time.

Every decoder below is a vectorized transcription of ggml's reference `dequantize_row_*` (ggml-quants.c).
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

# ggml type id -> (name, elements per block, bytes per block)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 30: ("BF16", 1, 2),
    2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
}
FLOAT_TYPES = (0, 1, 30)


def _u16(b, i):
    return b[..., i].astype(jnp.uint16) | (b[..., i + 1].astype(jnp.uint16) << 8)


def _f16(b, i):
    """Little-endian fp16 stored at byte offset i of every block -> f32 `[..., 1]`."""
    return lax.bitcast_convert_type(_u16(b, i), jnp.float16).astype(jnp.float32)[..., None]


def _shift(x, s):
    return (x >> jnp.asarray(s, dtype=x.dtype)) if not isinstance(s, int) else (x >> s)


# ---------------------------------------------------------------------- 32-element block formats
def _q8_0(b):
    q = lax.bitcast_convert_type(b[..., 2:34], jnp.int8).astype(jnp.float32)
    return _f16(b, 0) * q


def _q4_0(b):
    qs = b[..., 2:18]
    q = jnp.concatenate([qs & 15, qs >> 4], -1).astype(jnp.float32) - 8.0
    return _f16(b, 0) * q


def _q4_1(b):
    qs = b[..., 4:20]
    q = jnp.concatenate([qs & 15, qs >> 4], -1).astype(jnp.float32)
    return _f16(b, 0) * q + _f16(b, 2)


def _q5(b, off, signed):
    qh = (b[..., off].astype(jnp.uint32) | (b[..., off + 1].astype(jnp.uint32) << 8)
          | (b[..., off + 2].astype(jnp.uint32) << 16) | (b[..., off + 3].astype(jnp.uint32) << 24))[..., None]
    qs = b[..., off + 4:off + 20].astype(jnp.uint32)
    j = jnp.arange(16, dtype=jnp.uint32)
    x0 = (qs & 15) | (((qh >> j) << 4) & 16)
    x1 = (qs >> 4) | ((qh >> (j + 12)) & 16)
    q = jnp.concatenate([x0, x1], -1).astype(jnp.float32)
    return q - 16.0 if signed else q


def _q5_0(b):
    return _f16(b, 0) * _q5(b, 2, True)


def _q5_1(b):
    return _f16(b, 0) * _q5(b, 4, False) + _f16(b, 2)


# ---------------------------------------------------------------------- 256-element k-quants
def _q2_k(b):
    lead = b.shape[:-1]
    sc = b[..., 0:16].reshape(*lead, 2, 4, 2, 1)                       # is = h*8 + j*2 + p
    qs = b[..., 16:80].reshape(*lead, 2, 1, 2, 16)                     # byte = h*32 + p*16 + l
    sh = (2 * jnp.arange(4, dtype=jnp.uint8)).reshape(4, 1, 1)
    q = ((qs >> sh) & 3).astype(jnp.float32)                           # [..., h, j, p, l]
    d, dmin = _f16(b, 80)[..., None, None, None], _f16(b, 82)[..., None, None, None]
    y = d * (sc & 15).astype(jnp.float32) * q - dmin * (sc >> 4).astype(jnp.float32)
    return y.reshape(*lead, 256)


_K3_LO_IDX = np.array([k % 8 for k in range(16)])
_K3_LO_SH = np.array([4 * (k // 8) for k in range(16)], np.uint8)
_K3_HI_IDX = np.array([8 + k % 4 for k in range(16)])
_K3_HI_SH = np.array([2 * (k // 4) for k in range(16)], np.uint8)


def _q3_k(b):
    lead = b.shape[:-1]
    hm = b[..., 0:32].reshape(*lead, 1, 1, 2, 16)                      # byte = p*16 + l
    qs = b[..., 32:96].reshape(*lead, 2, 1, 2, 16)
    s = b[..., 96:108]
    lo = (s[..., _K3_LO_IDX] >> _K3_LO_SH) & 15
    hi = (s[..., _K3_HI_IDX] >> _K3_HI_SH) & 3
    sc = ((lo | (hi << 4)).astype(jnp.float32) - 32.0).reshape(*lead, 2, 4, 2, 1)
    jsh = (2 * jnp.arange(4, dtype=jnp.uint8)).reshape(4, 1, 1)
    q2 = ((qs >> jsh) & 3).astype(jnp.float32)
    bit = (jnp.arange(2, dtype=jnp.uint8)[:, None] * 4 + jnp.arange(4, dtype=jnp.uint8)[None, :]).reshape(2, 4, 1, 1)
    hbit = ((hm >> bit) & 1).astype(jnp.float32)                        # bit = h*4 + j
    q = q2 - 4.0 * (1.0 - hbit)
    return (_f16(b, 108)[..., None, None, None] * sc * q).reshape(*lead, 256)


def _k_scale_min(s):
    """get_scale_min_k4 for j = 0..7 -> (scale[8], min[8]) as f32."""
    lo_sc = s[..., 0:4] & 63
    lo_m = s[..., 4:8] & 63
    hi_sc = (s[..., 8:12] & 15) | ((s[..., 0:4] >> 6) << 4)
    hi_m = (s[..., 8:12] >> 4) | ((s[..., 4:8] >> 6) << 4)
    sc = jnp.concatenate([lo_sc, hi_sc], -1).astype(jnp.float32)
    m = jnp.concatenate([lo_m, hi_m], -1).astype(jnp.float32)
    return sc, m


def _q4_k(b):
    lead = b.shape[:-1]
    sc, m = _k_scale_min(b[..., 4:16])
    sc, m = sc.reshape(*lead, 4, 2, 1), m.reshape(*lead, 4, 2, 1)       # index 2g + p
    qs = b[..., 16:144].reshape(*lead, 4, 1, 32)
    q = ((qs >> jnp.array([0, 4], jnp.uint8).reshape(2, 1)) & 15).astype(jnp.float32)
    d, dmin = _f16(b, 0)[..., None, None], _f16(b, 2)[..., None, None]
    return (d * sc * q - dmin * m).reshape(*lead, 256)


def _q5_k(b):
    lead = b.shape[:-1]
    sc, m = _k_scale_min(b[..., 4:16])
    sc, m = sc.reshape(*lead, 4, 2, 1), m.reshape(*lead, 4, 2, 1)
    qh = b[..., 16:48].reshape(*lead, 1, 1, 32)
    qs = b[..., 48:176].reshape(*lead, 4, 1, 32)
    lo = (qs >> jnp.array([0, 4], jnp.uint8).reshape(2, 1)) & 15
    bit = (2 * jnp.arange(4, dtype=jnp.uint8)[:, None] + jnp.arange(2, dtype=jnp.uint8)[None, :]).reshape(4, 2, 1)
    hi = (qh >> bit) & 1
    q = (lo + 16 * hi).astype(jnp.float32)
    d, dmin = _f16(b, 0)[..., None, None], _f16(b, 2)[..., None, None]
    return (d * sc * q - dmin * m).reshape(*lead, 256)


def _q6_k(b):
    lead = b.shape[:-1]
    ql = b[..., 0:128].reshape(*lead, 2, 2, 32)                        # [h, r%2, l]
    qh = b[..., 128:192].reshape(*lead, 2, 1, 32)                      # [h, l]
    sc = lax.bitcast_convert_type(b[..., 192:208], jnp.int8).astype(jnp.float32).reshape(*lead, 2, 4, 2, 1)
    r = np.arange(4)
    low = (ql[..., r % 2, :] >> (4 * (r // 2)).astype(np.uint8).reshape(4, 1)) & 15   # [h, r, l]
    hi = (qh >> (2 * r).astype(np.uint8).reshape(4, 1)) & 3
    q = ((low | (hi << 4)).astype(jnp.float32) - 32.0).reshape(*lead, 2, 4, 2, 16)
    return (_f16(b, 208)[..., None, None, None] * sc * q).reshape(*lead, 256)


DEQUANT = {2: _q4_0, 3: _q4_1, 6: _q5_0, 7: _q5_1, 8: _q8_0, 10: _q2_k, 11: _q3_k, 12: _q4_k, 13: _q5_k, 14: _q6_k}


def dequant_blocks(blocks, qtype, out_dtype=jnp.bfloat16):
    """`blocks`: uint8 `[..., n_blocks, block_bytes]` -> `[..., n_blocks * block_elems]`."""
    y = DEQUANT[qtype](blocks)
    return y.reshape(*blocks.shape[:-2], -1).astype(out_dtype)


# ---------------------------------------------------------------------- host-side loading
class QWeight:
    """A linear weight `[out, in]` kept as raw GGML blocks (or a dense array for float types).

    Registered as a pytree so stacks of them can flow through `jit` / `lax.scan`.
    """

    def __init__(self, data, qtype, shape):
        self.data, self.qtype, self.shape = data, qtype, tuple(shape)

    def dense(self, dtype=jnp.bfloat16):
        if self.qtype in FLOAT_TYPES:
            return self.data.astype(dtype)
        w = dequant_blocks(self.data, self.qtype, dtype)
        return w.reshape(*self.data.shape[:-2], self.shape[-1]) if w.ndim >= 2 else w

    @property
    def nbytes(self):
        return int(np.prod(self.data.shape)) * self.data.dtype.itemsize


jax.tree_util.register_pytree_node(
    QWeight, lambda w: ((w.data,), (w.qtype, w.shape)), lambda aux, ch: QWeight(ch[0], *aux))


def linear(x, w, bias=None):
    """x `[..., in]` @ w.T -> `[..., out]` in f32, bf16 MXU inputs."""
    y = jnp.einsum("...i,oi->...o", x.astype(jnp.bfloat16), w.dense(jnp.bfloat16),
                   preferred_element_type=jnp.float32)
    return y if bias is None else y + bias.astype(jnp.float32)


def bf16_bits_to_f32(u16):
    return (np.asarray(u16, np.uint16).astype(np.uint32) << 16).view(np.float32)


def _on_tmpfs(path):
    """True if `path` lives on a tmpfs mount (longest matching mount point in /proc/mounts)."""
    import os
    path, best, fs = os.path.realpath(path), "", ""
    try:
        with open("/proc/mounts") as fh:
            for line in fh:
                _, mnt, typ = line.split()[:3]
                if (path == mnt or path.startswith(mnt.rstrip("/") + "/")) and len(mnt) > len(best):
                    best, fs = mnt, typ
    except OSError:
        return False
    return fs == "tmpfs"


class GGUFFile:
    """Thin wrapper over `gguf.GGUFReader` exposing tensors as host numpy arrays in torch layout."""

    def __init__(self, path):
        import gguf
        self.path = path
        self.reader = gguf.GGUFReader(path, "r")
        self.tensors = {t.name: t for t in self.reader.tensors}

    def preload(self, log=print, chunk=256 << 20):
        """Read the whole file once, sequentially, into host RAM. Colab's network disk serves scattered
        per-layer reads at ~13 MB/s but large sequential reads far faster; afterwards every tensor is
        sliced from memory. A file already in RAM (tmpfs, e.g. /dev/shm) is left memory-mapped."""
        import os, time
        if _on_tmpfs(self.path):
            log(f"      {os.path.basename(self.path)} is in RAM (tmpfs); using it memory-mapped")
            return self
        size = os.path.getsize(self.path)
        self.buf = np.empty(size, np.uint8)
        view, t0 = memoryview(self.buf), time.time()
        with open(self.path, "rb", buffering=0) as fh:
            got = 0
            while got < size:
                n = fh.readinto(view[got:got + chunk])
                if not n:
                    break
                got += n
        dt = time.time() - t0
        log(f"      read {os.path.basename(self.path)}: {size / 2**30:.2f} GiB in {dt:.1f}s ({size / 2**20 / max(dt, 1e-6):,.0f} MiB/s)")
        return self

    def _read_stack(self, names, row_shape, threads=8):
        """Stack the raw bytes of several equally-shaped tensors into one array (from the preloaded
        buffer when available, otherwise with parallel `readinto` calls)."""
        from concurrent.futures import ThreadPoolExecutor
        out = np.empty((len(names), *row_shape), np.uint8)
        buf = getattr(self, "buf", None)

        def one(i):
            t = self.tensors[names[i]]
            dst = out[i].reshape(-1)
            off = int(t.data_offset)
            if buf is not None:
                dst[:] = buf[off:off + dst.size]
                return
            view = memoryview(dst)
            with open(self.path, "rb", buffering=0) as fh:
                fh.seek(off)
                got = 0
                while got < len(view):
                    got += fh.readinto(view[got:])
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(one, range(len(names))))
        ref = np.asarray(self.tensors[names[-1]].data).reshape(-1)[:4096]
        assert np.array_equal(out[-1].reshape(-1)[:ref.size], ref), f"direct read mismatch for {names[-1]}"
        return out

    def qtype(self, name):
        return int(self.tensors[name].tensor_type)

    def host(self, name):
        """Raw host array. Float types: dense numpy (bf16 returned as uint16 bits). Quantized: uint8
        `[out, n_blocks, block_bytes]`."""
        t = self.tensors[name]
        qt = int(t.tensor_type)
        shape = [int(s) for s in reversed(t.shape)]                     # torch order [out, in]
        if qt in FLOAT_TYPES:
            arr = np.asarray(t.data)
            if qt == 30:
                arr = arr.view(np.uint16)
            return arr.reshape(shape)
        _, be, bb = GGML_TYPES[qt]
        return np.asarray(t.data).reshape(shape[0] if len(shape) > 1 else 1, shape[-1] // be, bb)

    def float32(self, name):
        a = self.host(name)
        return bf16_bits_to_f32(a) if self.qtype(name) == 30 else a.astype(np.float32)

    def qweight(self, name, put=jnp.asarray):
        qt = self.qtype(name)
        shape = [int(s) for s in reversed(self.tensors[name].shape)]
        if qt in FLOAT_TYPES:
            return QWeight(put(self.float32(name).astype(jnp.bfloat16)), 30, shape)
        return QWeight(put(self.host(name)), qt, shape)

    def stacked_qweight(self, names, put=jnp.asarray):
        """Stack the same tensor of several layers into one `QWeight` (needs identical types)."""
        types = {self.qtype(n) for n in names}
        assert len(types) == 1, f"mixed types {types} in {names[0]}"
        qt = types.pop()
        shape = [int(s) for s in reversed(self.tensors[names[0]].shape)]
        if qt in FLOAT_TYPES:
            arr = np.stack([self.float32(n) for n in names]).astype(jnp.bfloat16)
            return QWeight(put(arr), 30, shape)
        _, be, bb = GGML_TYPES[qt]
        return QWeight(put(self._read_stack(names, (shape[0], shape[-1] // be, bb))), qt, shape)
''',
    'h3/dit_jax.py': r'''"""MiniMax-H3 diffusion transformer (pruned GGUF form) in JAX.

Mirrors diffusers' `MiniMaxH3Transformer3DModel` + ComfyUI's pruned modulation (`adaln_t_table`).
The 50 blocks run under one `lax.scan`, so the block is compiled once and the quantized weights of every
block live stacked on the device.
"""
import json
import math
import struct
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from gguf_jax import QWeight, linear

HIDDEN, HEADS, HEAD_DIM, FFN = 5376, 56, 128, 14336
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2
ROPE_FRAME_RESCALE, ROPE_FRAMES_PER_LATENT, ROPE_SPATIAL_SCALE = 5.0 / 3.0, (1, 4, 4, 4, 4), 32
FPS, AUDIO_LATENTS_PER_SECOND = 24, 40
EPS = 1e-5


# ------------------------------------------------------------------------------------------ layout
def align_num_frames(n, frames_per_chunk=17, latents_per_chunk=5):
    while n % frames_per_chunk != latents_per_chunk:
        n += 1
    return n


def latent_frames(n, frames_per_chunk=17, latents_per_chunk=5):
    return (n - latents_per_chunk) // frames_per_chunk * latents_per_chunk + 2


def _spatial_grid(dim, patch, sqrt_area):
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return np.linspace(left, left + ratio, dim // patch, endpoint=False) * ROPE_SPATIAL_SCALE


def _temporal_grid(n, origin):
    spans = np.array([ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % 5] for i in range(n)], np.float64)
    return origin + np.concatenate([[0.0], np.cumsum(spans[:-1])])


def build_layout(num_text, num_latent_frames, latent_h, latent_w, num_audio_latents, text_tags=None):
    """`[text | audio ch0, ch1 | video]` packed layout of a t2va request (diffusers `build_packed_sequence`)."""
    rows_per_frame = (latent_h // 2) * (latent_w // 2)
    n_audio, n_video = 2 * num_audio_latents, num_latent_frames * rows_per_frame
    S = num_text + n_audio + n_video
    a0, v0 = num_text, num_text + n_audio
    pos = np.zeros((S, 3), np.float64)
    pos[:num_text, 0] = np.arange(num_text)
    sqrt_area = math.sqrt(latent_h * latent_w)
    hg, wg = _spatial_grid(latent_h, 2, sqrt_area), _spatial_grid(latent_w, 2, sqrt_area)
    frame = np.stack([g.reshape(-1) for g in np.meshgrid(hg, wg, indexing="ij")], -1)
    at = num_text + np.arange(num_audio_latents, dtype=np.float64)
    pos[a0:v0, 0] = np.tile(at, 2)
    pos[a0:v0, 2] = np.concatenate([np.full(num_audio_latents, wg[0]), np.full(num_audio_latents, wg[-1])])
    vp = np.empty((num_latent_frames, rows_per_frame, 3))
    vp[:, :, 0] = _temporal_grid(num_latent_frames, float(num_text))[:, None]
    vp[:, :, 1:] = frame[None]
    pos[v0:] = vp.reshape(-1, 3)
    tags = np.empty(S, np.int32)
    tags[:num_text] = TEXT_TAG if text_tags is None else np.asarray(text_tags)
    tags[a0:v0], tags[v0:] = AUDIO_TAG, VIDEO_TAG
    # timestep table rows: 0 = video t (text follows video), 1 = audio t
    t_index = np.zeros(S, np.int32)
    t_index[a0:v0] = 1
    return dict(position_ids=pos, adaln_idx=(t_index * 3 + tags).astype(np.int32), t_index=t_index,
                num_text=num_text, audio_start=a0, video_start=v0, seq_len=S)


def rope_cos_sin(position_ids, inv_freq):
    """`[S, 3]` -> cos/sin `[S, 96]` (t, h, w angle blocks, duplicated for the split-half rotation)."""
    pos = jnp.asarray(position_ids, jnp.float32)
    f = pos[:, :, None] * jnp.asarray(inv_freq, jnp.float32)[None, None, :]
    f = jnp.concatenate([f[:, 0], f[:, 1], f[:, 2]], -1)
    f = jnp.concatenate([f, f], -1)
    return jnp.cos(f), jnp.sin(f)


# ------------------------------------------------------------------------------------------ schedule
def schedule(num_points, shift):
    """MiniMaxH3Scheduler.set_timesteps: `num_points` sigma grid points incl. the terminal 0."""
    base = np.linspace(1.0, 0.0, num_points, dtype=np.float32)
    sig = (np.float32(shift) * base / (np.float32(1) + (np.float32(shift) - np.float32(1)) * base)).astype(np.float32)
    keep = np.concatenate([[True], sig[1:] != sig[:-1]])
    sig = sig[keep]
    return sig, (np.float32(1) - sig[:-1]).astype(np.float32)


def euler_step(x, v, t, sigma, sigma_next):
    denoised = x + (1.0 - t) * v
    r = sigma_next / sigma
    return r * x + (1.0 - r) * denoised


# ------------------------------------------------------------------------------------------ layers
def rms_norm(x, w=None, eps=EPS):
    x = x.astype(jnp.float32)
    y = x * lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + eps)
    return y if w is None else y * w.astype(jnp.float32)


def _rope(x, cos, sin):
    """Split-half rotation of the leading 96 channels of every head. x: `[S, H, D]`."""
    rd = cos.shape[-1]
    xr, xp = x[..., :rd], x[..., rd:]
    x1, x2 = xr[..., : rd // 2], xr[..., rd // 2:]
    rot = jnp.concatenate([-x2, x1], -1)
    c, s = cos[:, None, :], sin[:, None, :]
    return jnp.concatenate([xr * c + rot * s, xp], -1)


def chunked_attention(q, k, v, chunk=512):
    """Memory-bounded exact attention in plain XLA: query blocks of `chunk` rows, each against all keys
    (scores `[H, chunk, S]` f32 instead of `[H, S, S]`). Fallback when Pallas is unavailable."""
    S, H, Dh = q.shape
    pad = (-S) % chunk
    qb = jnp.pad(q, ((0, pad), (0, 0), (0, 0))).reshape(-1, chunk, H, Dh)

    def one(qc):
        s = jnp.einsum("qhd,khd->hqk", qc, k, preferred_element_type=jnp.float32) * (Dh ** -0.5)
        p = jax.nn.softmax(s, axis=-1)
        return jnp.einsum("hqk,khd->qhd", p.astype(jnp.bfloat16), v, preferred_element_type=jnp.float32).astype(q.dtype)

    return lax.map(one, qb).reshape(-1, H, Dh)[:S]


def attention(q, k, v, impl="xla"):
    """q/k/v `[S, H, D]` (bf16) -> `[S, H, D]`. Non-causal, single document."""
    if impl == "pallas":
        from attention_tpu import flash_attention_padded
        return flash_attention_padded(q, k, v)
    if impl == "chunked":
        return chunked_attention(q, k, v)
    return jax.nn.dot_product_attention(q[None], k[None], v[None], implementation="xla")[0]


def _lin(h, p, k):
    """`linear` plus the optional LoRA side path `(h A^T) B^T` (scale folded into B), kept low-rank so the
    quantized base weight is never re-materialized with the delta merged in."""
    y = linear(h, p[k])
    if k + "_la" in p:
        r = jnp.einsum("...i,ri->...r", h.astype(jnp.bfloat16), p[k + "_la"], preferred_element_type=jnp.float32)
        y = y + jnp.einsum("...r,or->...o", r.astype(jnp.bfloat16), p[k + "_lb"], preferred_element_type=jnp.float32)
    return y


def attn_block(h, p, cos=None, sin=None, impl="xla"):
    S = h.shape[0]
    qkv = _lin(h, p, "qkv")
    q, k, v = jnp.split(qkv, 3, -1)
    q = rms_norm(q.reshape(S, HEADS, HEAD_DIM), p["q_norm"])
    k = rms_norm(k.reshape(S, HEADS, HEAD_DIM), p["k_norm"])
    if cos is not None:
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
    o = attention(q.astype(jnp.bfloat16), k.astype(jnp.bfloat16), v.reshape(S, HEADS, HEAD_DIM).astype(jnp.bfloat16), impl)
    return _lin(o.reshape(S, HEADS * HEAD_DIM), p, "out")


def mlp_block(h, p):
    u = _lin(h, p, "fc1")
    gate, val = jnp.split(u, 2, -1)
    return _lin((jax.nn.silu(gate) * val).astype(jnp.bfloat16), p, "fc2")


def refiner_block(x, p, impl="xla"):
    x = x + attn_block(rms_norm(x, p["norm1"]), p, impl=impl)
    return x + mlp_block(rms_norm(x, p["norm2"]), p)


def dit_block(x, p, t_emb, idx, cos, sin, impl="xla"):
    mod = (t_emb @ p["ada_w"].astype(jnp.float32).T + p["ada_b"].astype(jnp.float32))   # [M, 3*6*D]
    mod = mod.reshape(-1, 6, HIDDEN)                                                   # [M*3, 6, D]
    sh1, sc1, g1, sh2, sc2, g2 = (mod[:, i] for i in range(6))
    h = rms_norm(x, p["norm1"]) * (1.0 + sc1[idx]) + sh1[idx]
    x = x + g1[idx] * attn_block(h, p, cos, sin, impl)
    h = rms_norm(x, p["norm2"]) * (1.0 + sc2[idx]) + sh2[idx]
    return x + g2[idx] * mlp_block(h, p)


def t_embedding(table, t_vals):
    """Pruned time embedding: linear interpolation in the `[1025, 8]` curve table."""
    pos = jnp.clip(t_vals, 0.0, 1.0) * (table.shape[0] - 1)
    i0 = jnp.minimum(jnp.floor(pos).astype(jnp.int32), table.shape[0] - 2)
    w = (pos - i0)[:, None]
    return table[i0] * (1.0 - w) + table[i0 + 1] * w


def encode_text(params, prompt_embeds, impl="xla"):
    """condition_proj + 2-block token refiner (text rows only; independent of the timestep)."""
    x = linear(prompt_embeds, params["cond_w"], params["cond_b"])
    x, _ = lax.scan(lambda c, p: (refiner_block(c, p, impl), None), x, params["refiner"])
    return rms_norm(x, params["refiner_final_norm"])


def forward(params, text, video_rows, audio_rows, t_v, t_a, layout, cos, sin, impl="xla"):
    """One denoiser evaluation over the packed sequence. Returns (video velocity, audio velocity), f32."""
    v_e = video_rows.astype(jnp.float32) @ params["video_in_w"].T + params["video_in_b"]
    a_e = audio_rows.astype(jnp.float32) @ params["audio_in_w"].T + params["audio_in_b"]
    x = jnp.concatenate([text.astype(jnp.float32), a_e, v_e], 0)
    t_emb = t_embedding(params["table"], jnp.stack([t_v, t_a]).astype(jnp.float32))
    idx = layout["adaln_idx"]

    def body(c, p):
        return dit_block(c, p, t_emb, idx, cos, sin, impl), None

    x, _ = lax.scan(body, x, params["blocks"])
    f = params["final"]
    mod = (t_emb @ f["ada_w"].T + f["ada_b"]).reshape(2, 2, HIDDEN)
    ti = layout["t_index"]
    h = rms_norm(x, f["norm"]) * (1.0 + mod[ti, 1]) + mod[ti, 0]
    a0, v0 = layout["audio_start"], layout["video_start"]
    video = h[v0:] @ f["video_w"].T + f["video_b"]
    audio = h[a0:v0] @ f["audio_w"].T + f["audio_b"]
    return video, audio


# ------------------------------------------------------------------------------------------ loading
_BLOCK_KEYS = {"norm1": "norm1.weight", "norm2": "norm2.weight", "q_norm": "attn.q_norm.weight",
               "k_norm": "attn.k_norm.weight", "qkv": "attn.qkv_proj.weight", "out": "attn.out_proj.weight",
               "fc1": "mlp.fc1.weight", "fc2": "mlp.fc2.weight"}


def load_dit(gguf, put=jnp.asarray, num_blocks=None, log=print):
    """Load the pruned DiT from a `GGUFFile`. Quantized matrices stay quantized (`QWeight`)."""
    f32 = lambda n: put(gguf.float32(n))  # noqa: E731
    nb = num_blocks or len({k.split(".")[1] for k in gguf.tensors if k.startswith("blocks.")})

    def stack(prefix, n, keys):
        import time
        out = {}
        for k, suffix in keys.items():
            t0 = time.time()
            names = [f"{prefix}.{i}.{suffix}" for i in range(n)]
            if k in ("norm1", "norm2", "q_norm", "k_norm", "ada_b", "ada_w"):
                out[k] = put(np.stack([gguf.float32(x) for x in names]))
                mb = out[k].nbytes / 2**20
            else:
                out[k] = gguf.stacked_qweight(names, put)
                mb = out[k].nbytes / 2**20
            dt = time.time() - t0
            if mb > 64:
                log(f"        {prefix}.*.{suffix}: {mb:,.0f} MiB in {dt:.1f}s ({mb / max(dt, 1e-6):,.0f} MiB/s)")
        return out

    params = dict(
        video_in_w=f32("video_patch_proj.weight"), video_in_b=f32("video_patch_proj.bias"),
        audio_in_w=f32("audio_patch_proj.weight"), audio_in_b=f32("audio_patch_proj.bias"),
        cond_w=gguf.qweight("condition_proj.weight", put), cond_b=f32("condition_proj.bias"),
        table=f32("adaln_t_table"), inv_freq=f32("rope.inv_freq"),
        refiner=stack("token_refiner.blocks", 2, _BLOCK_KEYS),
        refiner_final_norm=f32("token_refiner.final_norm.weight"),
    )
    log(f"      DiT globals + refiner loaded; loading {nb} blocks")
    params["blocks"] = stack("blocks", nb, dict(_BLOCK_KEYS, ada_w="adaln_proj.linear.weight",
                                                ada_b="adaln_proj.linear.bias"))
    params["final"] = dict(norm=f32("final_layer.norm.weight"), ada_w=f32("final_layer.adaln_proj.linear.weight"),
                           ada_b=f32("final_layer.adaln_proj.linear.bias"),
                           video_w=f32("final_layer.video_out.weight"), video_b=f32("final_layer.video_out.bias"),
                           audio_w=f32("final_layer.audio_out.weight"), audio_b=f32("final_layer.audio_out.bias"))
    return params


_LORA_KEYS = {"qkv": "attn.qkv_proj", "out": "attn.out_proj", "fc1": "mlp.fc1", "fc2": "mlp.fc2"}


def read_safetensors(path):
    """Minimal safetensors reader returning numpy views (bf16 via ml_dtypes, which jax ships with)."""
    import ml_dtypes
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        data = np.fromfile(fh, np.uint8)
    dts = {"BF16": ml_dtypes.bfloat16, "F16": np.float16, "F32": np.float32}
    meta = header.pop("__metadata__", {})
    out = {k: data[v["data_offsets"][0]:v["data_offsets"][1]].view(dts[v["dtype"]]).reshape(v["shape"])
           for k, v in header.items()}
    return out, meta


def add_lora(params, path, strength=1.0, put=jnp.asarray, log=print):
    """Attach a ComfyUI-format DiT LoRA (e.g. Comfy-Org's MiniMax-H3 turbo LoRAs) to `load_dit`'s params.

    The ComfyUI LoRA already uses the GGUF layout (`qkv_proj` = `[q; k; v]` with block-diagonal B,
    `fc1` = `[gate; value]`), so A/B pair with the quantized weights as is. delta W = B A * alpha / rank,
    rank = rows of A (ComfyUI's convention; the fused qkv alpha is pre-multiplied by 3 to match)."""
    t, meta = read_safetensors(path)
    nbytes = 0
    for group, prefix in (("blocks", "diffusion_model.blocks"), ("refiner", "diffusion_model.token_refiner.blocks")):
        p = params[group]
        n = p["qkv"].data.shape[0]
        for k, name in _LORA_KEYS.items():
            As, Bs = [], []
            for i in range(n):
                a, b = (t[f"{prefix}.{i}.{name}.lora_{x}.weight"] for x in "AB")
                scale = float(t[f"{prefix}.{i}.{name}.alpha"]) / a.shape[0] * strength
                assert a.shape[1] == p[k].shape[-1] and b.shape[0] == p[k].shape[0], (k, a.shape, b.shape, p[k].shape)
                As.append(a)
                Bs.append((b.astype(np.float32) * scale).astype(a.dtype))
            p[k + "_la"], p[k + "_lb"] = put(np.stack(As)), put(np.stack(Bs))
            nbytes += p[k + "_la"].nbytes + p[k + "_lb"].nbytes
    log(f"      LoRA {path.rsplit('/', 1)[-1]} x{strength} attached ({nbytes / 2**20:,.0f} MiB; "
        f"trained rank {meta.get('training_rank')}, alpha {meta.get('training_alpha')})")
    return params


def patchify(latents):
    """`[C, T, H, W]` -> rows `[T*(H/2)*(W/2), C*4]` (frame-major, row-major; channel-major patch)."""
    C, T, H, W = latents.shape
    x = latents.reshape(C, T, 1, H // 2, 2, W // 2, 2)
    return x.transpose(1, 3, 5, 0, 2, 4, 6).reshape(T * (H // 2) * (W // 2), C * 4)


def unpatchify(rows, C, T, H, W):
    x = rows.reshape(T, H // 2, W // 2, C, 1, 2, 2)
    return x.transpose(3, 0, 4, 1, 5, 2, 6).reshape(C, T, H, W)
''',
    'h3/te_jax.py': r'''"""MiniMax-H3 conditioner: Qwen3-VL-32B text stack truncated to 50 layers (GGUF), in JAX.

The conditioning is the hidden state after decoder layer 50 (no final norm). The prompt is tokenized
raw (no chat template, no special tokens). For text-only input Qwen3-VL's interleaved MRoPE gives every
axis the same position, i.e. plain 1-D rotate-half rope over the full head (theta 5e6).
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from gguf_jax import linear, bf16_bits_to_f32

HIDDEN, HEADS, KV_HEADS, HEAD_DIM = 5120, 64, 8, 128
EPS, THETA = 1e-6, 5_000_000.0


def rms_norm(x, w, eps=EPS):
    x = x.astype(jnp.float32)
    return x * lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + eps) * w.astype(jnp.float32)


def rope_cos_sin(n):
    inv = 1.0 / (THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    f = np.arange(n, dtype=np.float32)[:, None] * inv[None]
    f = np.concatenate([f, f], -1)
    return jnp.asarray(np.cos(f)), jnp.asarray(np.sin(f))


def _rope(x, cos, sin):
    half = HEAD_DIM // 2
    rot = jnp.concatenate([-x[..., half:], x[..., :half]], -1)
    return x * cos[:, None, :] + rot * sin[:, None, :]


def te_layer(x, p, cos, sin):
    L = x.shape[0]
    h = rms_norm(x, p["in_norm"])
    q = rms_norm(linear(h, p["q"]).reshape(L, HEADS, HEAD_DIM), p["q_norm"])
    k = rms_norm(linear(h, p["k"]).reshape(L, KV_HEADS, HEAD_DIM), p["k_norm"])
    v = linear(h, p["v"]).reshape(L, KV_HEADS, HEAD_DIM)
    q, k = _rope(q, cos, sin), _rope(k, cos, sin)
    o = jax.nn.dot_product_attention(q[None].astype(jnp.bfloat16), k[None].astype(jnp.bfloat16),
                                     v[None].astype(jnp.bfloat16), is_causal=True)[0]
    x = x + linear(o.reshape(L, HEADS * HEAD_DIM), p["o"])
    h = rms_norm(x, p["post_norm"])
    m = jax.nn.silu(linear(h, p["gate"])) * linear(h, p["up"])
    return x + linear(m.astype(jnp.bfloat16), p["down"])


@jax.jit
def te_forward(layers, x, cos, sin):
    x, _ = lax.scan(lambda c, p: (te_layer(c, p, cos, sin), None), x.astype(jnp.float32), layers)
    return x


_KEYS = {"in_norm": "input_layernorm.weight", "post_norm": "post_attention_layernorm.weight",
         "q_norm": "self_attn.q_norm.weight", "k_norm": "self_attn.k_norm.weight",
         "q": "self_attn.q_proj.weight", "k": "self_attn.k_proj.weight", "v": "self_attn.v_proj.weight",
         "o": "self_attn.o_proj.weight", "gate": "mlp.gate_proj.weight", "up": "mlp.up_proj.weight",
         "down": "mlp.down_proj.weight"}


def load_te_layers(gguf, put=jnp.asarray, layers=None, log=print):
    import time
    n = len({k.split(".")[2] for k in gguf.tensors if k.startswith("model.layers.")})
    idx = list(range(n)) if layers is None else list(layers)
    out = {}
    for k, suffix in _KEYS.items():
        t0 = time.time()
        names = [f"model.layers.{i}.{suffix}" for i in idx]
        if k.endswith("norm"):
            out[k] = put(np.stack([gguf.float32(x) for x in names]))
        else:
            out[k] = gguf.stacked_qweight(names, put)
            mb, dt = out[k].nbytes / 2**20, time.time() - t0
            log(f"        layers.*.{suffix}: {mb:,.0f} MiB in {dt:.1f}s ({mb / max(dt, 1e-6):,.0f} MiB/s)")
    return out


def embed(gguf, token_ids):
    """Host-side row gather from the bf16 embedding table (avoids shipping 1.5 GB to the device)."""
    table = gguf.host("model.embed_tokens.weight")                      # uint16 bf16 bits [V, H]
    return bf16_bits_to_f32(table[np.asarray(token_ids)])
''',
    'h3/vae_jax.py': r'''"""MiniMax-H3 decoders in JAX: ViT video decoder (Comfy-format weights) and BigVGAN audio decoder.

Video: diffusers `AutoencoderKLMiniMaxH3._decode` chunking (5 latent frames + 2 overlap, cross-faded) and
256 px spatial tiles (linear blends), with every tile of a chunk decoded in one batched call.
Audio: fp32 throughout (the checkpoint degrades audibly in bf16).
"""
import math
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from gguf_jax import linear, QWeight

VDIM, VHEADS, VHD, VREG = 2048, 32, 64, 4
AUDIO_PRECISION = {"p": lax.Precision.HIGHEST}   # conv precision of the BigVGAN decoder (set before first call)


def _hi():
    return AUDIO_PRECISION["p"]


# ======================================================================================= video
def _rms(x, w=None, eps=1e-5):
    x = x.astype(jnp.float32)
    y = x * lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + eps)
    return y if w is None else y * w


def vit_rope(T, H, W):
    """diffusers `MiniMaxH3VideoRotaryPosEmbed` over the token grid + 5 zero suffix tokens -> [N, 48]."""
    dim, axes, theta = int(VHD * 0.75), 3, 100.0
    inv = 1.0 / theta ** np.arange(0, 1, 2 * axes / dim, dtype=np.float32)
    grids = [2.0 * (np.arange(0.5, s, dtype=np.float32) / s) - 1.0 for s in (T, H, W)]
    pos = np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(-1, 3)
    pos = np.concatenate([pos, np.zeros((VREG + 1, 3), np.float32)], 0)
    ang = (2.0 * math.pi * pos[:, :, None] * inv[None, None]).reshape(pos.shape[0], -1)
    ang = np.concatenate([ang, ang], -1)
    return jnp.asarray(np.cos(ang)), jnp.asarray(np.sin(ang))


def _vit_block(x, p, cos, sin):
    B, N, _ = x.shape
    h = _rms(x, p["norm1"])
    qkv = linear(h, p["qkv"], p["qkv_b"]).reshape(B, N, VHEADS, 3 * VHD)       # per-head [q k v]
    q, k, v = jnp.split(qkv, 3, -1)
    q, k = _rms(q), _rms(k)
    rd = cos.shape[-1]
    c, s = cos[None, :, None, :], sin[None, :, None, :]

    def rope(t):
        tr, tp = t[..., :rd], t[..., rd:]
        rot = jnp.concatenate([-tr[..., rd // 2:], tr[..., : rd // 2]], -1)
        return jnp.concatenate([tr * c + rot * s, tp], -1)

    q, k = rope(q), rope(k)
    o = jax.nn.dot_product_attention(q.astype(jnp.bfloat16), k.astype(jnp.bfloat16), v.astype(jnp.bfloat16))
    o = jnp.nan_to_num(o.astype(jnp.float32)).reshape(B, N, VDIM)
    x = x + linear(o, p["out"], p["out_b"]) * p["scale1"]
    h = _rms(x, p["norm2"])
    gate, val = jnp.split(linear(h, p["w1"], p["w1_b"]), 2, -1)
    return x + linear((jax.nn.silu(gate) * val).astype(jnp.bfloat16), p["w2"], p["w2_b"]) * p["scale2"]


@jax.jit
def vit_decode(params, z):
    """`z`: `[B, 24, T, H, W]` (after post_quant_conv) -> pixels `[B, 3, 4T, 16H, 16W]` (ImageNet-normalized)."""
    B, C, T, H, W = z.shape
    x = z.transpose(0, 2, 3, 4, 1).reshape(B, T * H * W, C).astype(jnp.float32)
    x = x @ params["x_emb_w"].T + params["x_emb_b"]
    n = x.shape[1]
    x = jnp.concatenate([x, jnp.broadcast_to(params["register"], (B, VREG, VDIM)), jnp.zeros((B, 1, VDIM))], 1)
    cos, sin = vit_rope(T, H, W)
    x, _ = lax.scan(lambda c, p: (_vit_block(c, p, cos, sin), None), x, params["blocks"])
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    x = (x - mu) * lax.rsqrt(var + 1e-5) * params["norm_out_w"] + params["norm_out_b"]
    y = linear(x[:, :n], params["proj_out"], params["proj_out_b"])
    y = y.reshape(B, T, H, W, 3, 4, 16, 16).transpose(0, 4, 1, 5, 2, 6, 3, 7)
    return y.reshape(B, 3, T * 4, H * 16, W * 16)


def post_quant_conv(params, z):
    return jnp.einsum("oc,bcthw->bothw", params["pq_w"], z) + params["pq_b"][None, :, None, None, None]


def load_video_vae(path, put=jnp.asarray, num_blocks=None):
    from safetensors import safe_open
    with safe_open(path, "np") as fh:
        keys = [k for k in fh.keys() if not k.startswith("encoder.")]
        pre = "decoder.transformer_blocks"
        n = num_blocks or len({k.split(".")[2] for k in keys if k.startswith(pre)})
        keys = [k for k in keys if not k.startswith(pre) or int(k.split(".")[2]) < n]
        sd = {k: fh.get_tensor(k) for k in keys}
    f32 = lambda k: put(sd[k].astype(np.float32))  # noqa: E731
    bf = lambda k: QWeight(put(sd[k].astype(jnp.bfloat16)), 30, sd[k].shape)  # noqa: E731

    def stk(suffix, conv):
        arr = np.stack([sd[f"{pre}.{i}.{suffix}"] for i in range(n)])
        if conv == "w":
            return QWeight(put(arr.astype(jnp.bfloat16)), 30, arr.shape[1:])
        return put(arr.astype(np.float32))

    blocks = {"norm1": stk("norm1.weight", "f"), "norm2": stk("norm2.weight", "f"),
              "scale1": stk("scale1", "f"), "scale2": stk("scale2", "f"),
              "qkv": stk("attn.to_qkv.weight", "w"), "qkv_b": stk("attn.to_qkv.bias", "f"),
              "out": stk("attn.to_out.weight", "w"), "out_b": stk("attn.to_out.bias", "f"),
              "w1": stk("ff.w1.weight", "w"), "w1_b": stk("ff.w1.bias", "f"),
              "w2": stk("ff.w2.weight", "w"), "w2_b": stk("ff.w2.bias", "f")}
    params = dict(blocks=blocks, x_emb_w=f32("decoder.x_embedder.weight"), x_emb_b=f32("decoder.x_embedder.bias"),
                  register=f32("decoder.register_tokens"), norm_out_w=f32("decoder.norm_out.weight"),
                  norm_out_b=f32("decoder.norm_out.bias"), proj_out=bf("decoder.proj_out.weight"),
                  proj_out_b=f32("decoder.proj_out.bias"),
                  pq_w=put(sd["post_quant_conv.weight"].reshape(24, 24).astype(np.float32)),
                  pq_b=f32("post_quant_conv.bias"))
    stats = (sd["latents_mean"].astype(np.float32), sd["latents_std"].astype(np.float32))
    return params, stats


# ----- diffusers `_decode` / `_decode_clip` geometry (numpy, host side) -----
SPATIAL, TEMPORAL, CLIP, TOKEN_DROP = 16, 4, 17, 3
TOKENS_CHUNK = math.ceil(CLIP / TEMPORAL)                 # 5
TOKEN_OVERLAP = (-TOKEN_DROP) % TOKENS_CHUNK               # 2
FRAME_PRE_PAD = (-CLIP) % TEMPORAL                         # 3
FRAME_OVERLAP = max(TOKEN_OVERLAP * TEMPORAL - FRAME_PRE_PAD, 0)   # 5


def split_tiles(length, tile=256, min_overlap=64):
    if tile >= length:
        return [0], [length], []
    n = math.ceil(length / tile)
    while tile * n - min_overlap * (n - 1) - length < 0:
        n += 1
    ov = [min_overlap] * (n - 1)
    rem = tile * n - sum(ov) - length
    for i in range(rem // SPATIAL):
        ov[i % (n - 1)] += SPATIAL
    st = [0]
    for i in range(n - 1):
        st.append(st[-1] + tile - ov[i])
    return st, [tile] * n, ov


def _blend(a, b, ext, axis):
    ext = min(a.shape[axis], b.shape[axis], ext)
    w = np.arange(ext, dtype=np.float32) / ext
    shp = [1] * a.ndim
    shp[axis] = ext
    w = w.reshape(shp)
    sa = [slice(None)] * a.ndim; sa[axis] = slice(-ext, None)
    sb = [slice(None)] * b.ndim; sb[axis] = slice(0, ext)
    blended = a[tuple(sa)] * (1 - w) + b[tuple(sb)] * w
    if ext == b.shape[axis]:
        return blended
    sr = [slice(None)] * b.ndim; sr[axis] = slice(ext, None)
    return np.concatenate([blended, b[tuple(sr)]], axis)


def decode_clip(z, decode_tiles, tile=256, overlap=64):
    """z `[1, 24, t, H, W]` (numpy) -> pixels; all tiles decoded in one batched `decode_tiles` call."""
    H, W = z.shape[-2] * SPATIAL, z.shape[-1] * SPATIAL
    ys, yl, yo = split_tiles(H, tile, overlap)
    xs, xl, xo = split_tiles(W, tile, overlap)
    crops = [z[..., i // SPATIAL:(i + li) // SPATIAL, j // SPATIAL:(j + lj) // SPATIAL]
             for i, li in zip(ys, yl) for j, lj in zip(xs, xl)]
    out = decode_tiles(np.concatenate(crops, 0))
    rows = [[out[r * len(xs) + c:r * len(xs) + c + 1] for c in range(len(xs))] for r in range(len(ys))]
    result = []
    for i, row in enumerate(rows):
        rr = []
        for j, t in enumerate(row):
            if i > 0:
                t = _blend(rows[i - 1][j], t, yo[i - 1], -2)
            if j > 0:
                t = _blend(row[j - 1], t, xo[j - 1], -1)
            if i < len(rows) - 1:
                t = t[..., :-yo[i], :]
            if j < len(row) - 1:
                t = t[..., :, :-xo[j]]
            rr.append(t)
        result.append(np.concatenate(rr, -1))
    return np.concatenate(result, -2)


def decode_video(z, decode_tiles, tile=256, overlap=64):
    """diffusers `_decode`: temporal chunks of 5 latent frames (+2 overlap), cross-faded."""
    n_tok = z.shape[2] + TOKEN_DROP
    pad = (-n_tok) % TOKENS_CHUNK
    n_chunks = (n_tok + pad) // TOKENS_CHUNK - int(TOKEN_DROP > 0)
    if pad > 0:
        z = np.concatenate([z, np.repeat(z[:, :, -1:], pad, 2)], 2)
    chunk_frames = TOKENS_CHUNK * TEMPORAL
    decoded, overlap_chunk = [], None
    for i in range(n_chunks):
        s = i * TOKENS_CHUNK
        clip = decode_clip(z[:, :, s:s + TOKENS_CHUNK + TOKEN_OVERLAP], decode_tiles, tile, overlap)
        for j in range(int(TOKEN_DROP > 0) + 1):
            ch = clip[:, :, j * chunk_frames:(j + 1) * chunk_frames][:, :, FRAME_PRE_PAD:]
            if j == 0:
                if overlap_chunk is not None:
                    ch = _blend(overlap_chunk, ch, FRAME_OVERLAP, -3)
                decoded.append(ch)
            else:
                overlap_chunk = ch
    if overlap_chunk is not None:
        decoded.append(overlap_chunk)
    dec = np.concatenate(decoded, 2)
    if pad > 0:
        tail = CLIP % TEMPORAL
        before = z.shape[2] - pad
        drop = sum(tail if tail and (before + k) % TOKENS_CHUNK == 0 else TEMPORAL for k in range(pad))
        dec = dec[:, :, :-drop]
    return dec


# ======================================================================================= audio
def _conv1d(x, w, b=None, stride=1, dilation=1, pad=(0, 0), groups=1):
    y = lax.conv_general_dilated(x, w, (stride,), [pad], rhs_dilation=(dilation,),
                                 dimension_numbers=("NCH", "OIH", "NCH"), feature_group_count=groups, precision=_hi())
    return y if b is None else y + b[None, :, None]


def _conv_t1d(x, w, b, stride, padding, groups=1):
    """torch ConvTranspose1d (weight `[in, out/groups, K]`) via an input-dilated convolution."""
    K = w.shape[-1]
    if groups == 1:
        wk = jnp.flip(w, -1).transpose(1, 0, 2)
    else:
        wk = jnp.flip(w, -1)                                    # depthwise: [C, 1, K] already O x I/g
    y = lax.conv_general_dilated(x, wk, (1,), [(K - 1 - padding, K - 1 - padding)], lhs_dilation=(stride,),
                                 dimension_numbers=("NCH", "OIH", "NCH"), feature_group_count=groups, precision=_hi())
    return y if b is None else y + b[None, :, None]


def _pad_rep(x, l, r):
    return jnp.pad(x, ((0, 0), (0, 0), (l, r)), mode="edge")


def _upsample2(x, f):
    """Exact polyphase form of `pad_rep(5,5) -> 2 * conv_transpose(f, stride 2) -> crop [15:-15]` with a
    12-tap filter shared by all channels: two 6-tap FIRs of shifted slices (no depthwise conv, which
    XLA compiles and runs slowly on TPU)."""
    L = x.shape[-1]
    xp = _pad_rep(x, 5, 5)
    f = f.reshape(-1)
    even = sum(f[2 * i + 1] * xp[..., 7 - i:7 - i + L] for i in range(6))   # out[2u]
    odd = sum(f[2 * i] * xp[..., 8 - i:8 - i + L] for i in range(6))        # out[2u+1]
    return 2.0 * jnp.stack([even, odd], -1).reshape(*x.shape[:-1], 2 * L)


def _downsample2(y, g):
    """Exact `pad_rep(5,6) -> conv(g, stride 2)` (12 taps) as 12 strided slices."""
    L2 = y.shape[-1]
    z = _pad_rep(y, 5, 6)
    g = g.reshape(-1)
    return sum(g[t] * z[..., t:t + L2 - 1:2] for t in range(12))


def _act1d(x, p):
    """Alias-free SnakeBeta: 2x Kaiser upsample -> SnakeBeta -> 2x Kaiser low-pass downsample."""
    y = _upsample2(x, p["up"])
    a, bt = jnp.exp(p["alpha"])[None, :, None], jnp.exp(p["beta"])[None, :, None]
    y = y + (1.0 / (bt + 1e-9)) * jnp.sin(a * y) ** 2
    # without the barrier XLA fuses the 12 strided reads of the downsampler into the 12-slice upsampler
    # (producer duplication), which blew the compile up past 46 GB of host RAM
    y = lax.optimization_barrier(y)
    return lax.optimization_barrier(_downsample2(y, p["down"]))


RATES, KSIZES, RES_K = (5, 5, 2, 2, 2, 2, 2), (9, 9, 4, 4, 4, 4, 4), (3, 7, 11)

# One fully-unrolled BigVGAN graph took > 4 min (and once > 46 GB host RAM) to compile on TPU. Instead
# every repeated unit is its own small jitted program, compiled once per (shape, static config) and reused:
# ~8 activation shapes + ~90 single-conv programs, each compiling in well under a second.
_act_j = jax.jit(_act1d)
_conv_j = jax.jit(_conv1d, static_argnames=("stride", "dilation", "pad", "groups"))
_convt_j = jax.jit(_conv_t1d, static_argnames=("stride", "padding", "groups"))


@jax.jit
def _avg3(a, b, c):
    return (a + b + c) / 3.0


@jax.jit
def _add(a, b):
    return a + b


def _amp(x, p, k):
    for i, d in enumerate((1, 3, 5)):
        r = _conv_j(_act_j(x, p["acts"][2 * i]), p["c1"][i]["w"], p["c1"][i]["b"], dilation=d,
                    pad=((k * d - d) // 2,) * 2)
        r = _conv_j(_act_j(r, p["acts"][2 * i + 1]), p["c2"][i]["w"], p["c2"][i]["b"], pad=((k - 1) // 2,) * 2)
        x = _add(x, r)
    return x


def audio_decode(p, lat):
    """lat `[B, 32, T]` (denormalized) -> waveform `[B, 1, T*800]` in [-1, 1]."""
    x = _conv_j(jnp.asarray(lat, jnp.float32), p["dec_in_w"], p["dec_in_b"])
    x = _conv_j(x, p["pre_w"], p["pre_b"], pad=(3, 3))
    for i, (r, k) in enumerate(zip(RATES, KSIZES)):
        x = _convt_j(x, p["ups"][i]["w"], p["ups"][i]["b"], stride=r, padding=(k - r) // 2)
        x = _avg3(*(_amp(x, p["res"][i * 3 + j], rk) for j, rk in enumerate(RES_K)))
    x = _act_j(x, p["post_act"])
    x = _conv_j(x, p["post_w"], None, pad=(3, 3))
    return jnp.clip(x, -1.0, 1.0)


def load_audio_vae(path, put=jnp.asarray):
    from safetensors.numpy import load_file
    sd = {k: v.astype(np.float32) for k, v in load_file(path).items()}
    a = lambda k: put(sd[k])  # noqa: E731

    def act(pre):
        return {"alpha": a(pre + ".act.alpha"), "beta": a(pre + ".act.beta"),
                "up": a(pre + ".upsample.filter"), "down": a(pre + ".downsample.lowpass.filter")}

    res = []
    for r in range(21):
        pre = f"decoder.resblocks.{r}"
        res.append({"acts": [act(f"{pre}.activations.{i}") for i in range(6)],
                    "c1": [{"w": a(f"{pre}.convs1.{i}.weight"), "b": a(f"{pre}.convs1.{i}.bias")} for i in range(3)],
                    "c2": [{"w": a(f"{pre}.convs2.{i}.weight"), "b": a(f"{pre}.convs2.{i}.bias")} for i in range(3)]})
    p = dict(dec_in_w=a("dec_in_proj.weight"), dec_in_b=a("dec_in_proj.bias"),
             pre_w=a("decoder.conv_pre.weight"), pre_b=a("decoder.conv_pre.bias"),
             ups=[{"w": a(f"decoder.ups.{i}.0.weight"), "b": a(f"decoder.ups.{i}.0.bias")} for i in range(7)],
             res=res, post_act=act("decoder.activation_post"), post_w=a("decoder.conv_post.weight"))
    return p, (sd["latents_mean"], sd["latents_std"])
''',
    'h3/attention_tpu.py': r'''"""TPU attention: Pallas flash attention over the padded packed sequence.

The packed MiniMax-H3 sequence (text + audio + video rows) is one attention document of arbitrary length.
It is padded up to the kernel block size and the pad rows get their own segment id, so real rows never
attend to padding (and pad rows attend only to each other, which keeps them NaN-free).
"""
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu import flash_attention as fa

BLOCK = 512


def _block_sizes(b):
    return fa.BlockSizes(block_q=b, block_k_major=b, block_k=b, block_b=1,
                         block_q_major_dkv=b, block_k_major_dkv=b, block_k_dkv=b, block_q_dkv=b,
                         block_k_major_dq=b, block_k_dq=b, block_q_dq=b)


def flash_attention_padded(q, k, v, block=BLOCK):
    """q/k/v `[S, H, D]` -> `[S, H, D]`, non-causal, softmax scale 1/sqrt(D)."""
    S, H, D = q.shape
    pad = (-S) % block

    def prep(x):
        return jnp.pad(x, ((0, pad), (0, 0), (0, 0))).transpose(1, 0, 2)[None]

    seg = jnp.concatenate([jnp.ones((S,), jnp.int32), jnp.zeros((pad,), jnp.int32)])[None]
    o = fa.flash_attention(prep(q), prep(k), prep(v), segment_ids=fa.SegmentIds(seg, seg),
                           sm_scale=D ** -0.5, block_sizes=_block_sizes(block))
    return o[0].transpose(1, 0, 2)[:S]
''',
}


def nzap(event, console, **fields):
    """One NZAP app event; `console` is what the console shows."""
    payload = {"v": 1, "app": APP, "event": event, **fields}
    display({"application/vnd.nzap.app+json": payload, "text/plain": console}, raw=True)


def stage(stage_id, label, progress=None):
    fields = {"id": stage_id, "label": label}
    if progress is not None:
        fields["progress"] = round(progress, 3)
    nzap("stage", f"[nzap] {label}", **fields)


def pip_version(package):
    out = subprocess.run([sys.executable, "-m", "pip", "show", package], capture_output=True, text=True).stdout
    match = re.search(r"^Version: (\S+)", out, re.M)
    return match.group(1) if match else None


started = time.time()
if not (glob.glob("/dev/accel*") or os.path.exists("/dev/vfio/0")):
    raise RuntimeError("MiniMax-H3 needs a TPU runtime: create a TPU v5e-1 runtime and run it there.")
size = params.get("size", "672x384")
width, height = (int(v) for v in size.split("x"))
prompt = params["prompt"].strip()
if not prompt:
    raise ValueError("Describe the video you want, including what it should sound like.")

# The weights are warm when the two GGUFs are still in RAM from an earlier run on this runtime.
warm = len(glob.glob(f"{RAM_DIR}/*.gguf")) >= 2

if (pip_version("jax"), pip_version("libtpu"), pip_version("gguf")) != (JAX_VERSION, LIBTPU_VERSION, GGUF_VERSION):
    stage("install", "Installing JAX for TPU")
    # libtpu must match jax: a mismatch breaks the Pallas attention kernel ("expected <= 7 but got 8").
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"jax[tpu]=={JAX_VERSION}",
                           f"libtpu=={LIBTPU_VERSION}", f"gguf=={GGUF_VERSION}"])

code_dir = f"{ROOT}/code"
for name, text in SOURCES.items():
    path = os.path.join(code_dir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path) or open(path, encoding="utf-8").read() != text:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)

os.makedirs(OUT_DIR, exist_ok=True)
output = f"{OUT_DIR}/minimax_h3_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
cmd = [sys.executable, "-u", "inference_minimax_h3.py", "--prompt", prompt, "--turbo", params.get("steps", "8"),
       "--width", str(width), "--height", str(height), "--seed", str(int(params.get("seed", 0))),
       "--lora_strength", str(float(params.get("lora_strength", 1.0))),
       "--weights_dir", f"{ROOT}/weights", "--cache_dir", f"{ROOT}/cache", "--ram_dir", RAM_DIR, "--output", output]

stage("download", "Checking weights" if warm else "Downloading weights (~28 GB, first run only)")
proc = subprocess.Popen(cmd, cwd=code_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
setup_seconds, run_started, tail, timings = None, None, [], {}
for line in proc.stdout:
    print(line, end="", flush=True)
    tail = (tail + [line.rstrip()])[-25:]
    t = re.match(r"\[\s*([\d.]+)s\]", line)
    t = float(t.group(1)) if t else None
    if "weights ready" in line:
        setup_seconds = time.time() - started
        run_started = time.time()
        nzap("ready", "[nzap] Weights ready.", warm=warm, setupSeconds=round(setup_seconds, 2), device="TPU v5e")
        stage("load", "Loading the text encoder")
    elif "[1/3] text encoder loaded" in line or "prompt embeddings from cache" in line:
        timings["text encoder"] = t
        stage("load", "Encoding the prompt")
    elif "[1/3] prompt encoded" in line:
        stage("load", "Loading the video model and turbo LoRA")
    elif "[2/3] DiT" in line and "loaded" in line:
        timings["video model"] = t
        stage("run", "Generating (compiling the first step)", 0.0)
    elif m := re.search(r"\[2/3\] step (\d+)/(\d+)", line):
        i, n = int(m.group(1)), int(m.group(2))
        if i == n:
            timings["generation"] = t
            stage("decode", "Decoding video and audio")
        else:
            stage("run", f"Generating: step {i + 1} of {n}", i / n)
    elif "[2/3] latents from cache" in line:
        stage("decode", "Decoding video and audio")
    elif "[3/3] audio decoded" in line:
        timings["decode"] = t
rc = proc.wait()
if rc != 0 or not os.path.exists(output):
    text = "\n".join(tail)
    if "RESOURCE_EXHAUSTED" in text or "out of memory" in text.lower():
        raise RuntimeError("The TPU ran out of memory: use 672x384 or a v6e-1 runtime.")
    raise RuntimeError(f"MiniMax-H3 failed (exit code {rc}); the last log lines are printed above.")

run_seconds = time.time() - (run_started or started)
nzap("output", f"[nzap] Saved {output}", id="video", kind="video", path=output, mime="video/mp4")
rows = [["Prompt", prompt], ["Size", f"{width}x{height}, 124 frames at 24 fps (5.2 s) + 32 kHz stereo"],
        ["Steps", params.get("steps", "8")], ["Seed", int(params.get("seed", 0))]]
rows += [[f"{k.capitalize()} done at", f"{v:.0f} s"] for k, v in timings.items() if v is not None]
nzap("output", "[nzap] Details", id="details", kind="table", columns=["Setting", "Value"], rows=rows)
nzap("done", f"[nzap] Done in {time.time() - started:.1f}s.",
     seconds={"setup": round(setup_seconds or 0, 2), "run": round(run_seconds, 2)}, warm=warm)
