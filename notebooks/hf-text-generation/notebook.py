# NZAP Engine injects a `params` dict before this script runs.
# Generates text with a Hugging Face model on this runtime (GPU if present).

import subprocess
import sys

try:
    import transformers  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers"])

import torch
from transformers import pipeline

device = 0 if torch.cuda.is_available() else -1
generator = pipeline("text-generation", model=params["model"], device=device)
result = generator(
    params["prompt"],
    max_new_tokens=params["max_new_tokens"],
    do_sample=params["temperature"] > 0,
    temperature=max(params["temperature"], 1e-5),
)
print(result[0]["generated_text"])
