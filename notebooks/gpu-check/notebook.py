# NZAP Engine injects a `params` dict before this script runs.
# Reports the accelerator attached to this runtime, as each framework sees it.

import shutil
import subprocess

if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
else:
    print("nvidia-smi not found: this runtime has no NVIDIA GPU.")

try:
    import torch

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            print(f"torch sees cuda:{index} {props.name} ({props.total_memory / 2**30:.1f} GiB)")
    else:
        print("torch: CUDA is not available")
except ImportError:
    print("torch is not installed")

if params.get("check_jax"):
    try:
        import jax

        print(f"jax devices: {jax.devices()}")
    except ImportError:
        print("jax is not installed")
