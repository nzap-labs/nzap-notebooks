# NZAP Engine injects a `params` dict before this script runs.
# Classifies the sentiment of one or more lines of text.

import subprocess
import sys

try:
    import transformers  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers"])

import torch
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if torch.cuda.is_available() else -1,
)
lines = [line.strip() for line in params["text"].splitlines() if line.strip()]
for line, result in zip(lines, classifier(lines)):
    print(f"{result['label']:>8} {result['score']:.3f}  {line}")
