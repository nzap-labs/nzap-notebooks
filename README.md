# NZAP public notebooks

The public notebook collection for [NZAP Engine](https://github.com/nzap-labs/nzap-engine).
The app reads `index.json` from this repository, downloads a notebook's source
when you open or run it, and refuses anything whose SHA-256 does not match the
index. Notebooks run on the user's own Colab runtime, never on their computer.

## Layout

```
index.json                    generated catalog (do not edit by hand)
notebooks/<slug>/notebook.py  the source that runs
notebooks/<slug>/notebook.json metadata and the parameter schema
notebooks/<slug>/app.json     optional: turns the notebook into an app (APPS.md)
scripts/build_index.py        builds and validates index.json
```

## How a notebook gets its parameters

NZAP Engine injects a `params` dict ahead of the source, built from the values
the user entered and validated against the declared schema:

```python
print(params["string_to_print"])
```

`notebook.json` declares those parameters:

```json
{
  "slug": "print-notebook",
  "title": "Print Notebook",
  "description": "Prints whatever string you give it.",
  "tags": ["starter"],
  "author": "Your Name",
  "params": [
    {
      "key": "string_to_print",
      "label": "String to print",
      "type": "string",
      "default": "Hello from NZAP",
      "required": true
    }
  ]
}
```

Parameter `type` is one of `string`, `text`, `integer`, `number`, `boolean` or
`select`. A `select` also needs `options`.

## Apps

A notebook with an `app.json` beside it becomes an **app** in NZAP Engine: a
form with real widgets (sliders, voice pickers, file uploads), the runtime it
needs, how long setup and each run take, and results rendered as audio,
images or tables instead of console text. [APPS.md](./APPS.md) is the format.

| App                                                     | What it does                                              | Runtime |
| ------------------------------------------------------- | --------------------------------------------------------- | ------- |
| [Kokoro Text to Speech](./notebooks/kokoro-tts)         | 54 voices in nine languages from an 82M model             | CPU, T4 |
| [Sentiment Analysis](./notebooks/hf-sentiment)          | Positive / negative scores per line, as a table           | CPU     |
| [Breeze TTS 2](./notebooks/breeze-tts)                  | Voice design in plain words, or cloning from a clip (non-commercial weights) | T4+     |
| [Z-Image](./notebooks/z-image)                          | Photorealistic text to image from a GGUF 6B model (Apache-2.0) | T4+     |
| [Qwen-Image 2.1](./notebooks/qwen-image)                | Text to image with legible in-image text (non-commercial weights) | T4+     |
| [Media Downloader](./notebooks/media-downloader)        | Video or audio from YouTube and 1,800+ sites (yt-dlp), with trimming | CPU     |
| [Background Remover](./notebooks/background-remover)    | Transparent cut-outs, new colour or blurred background (BiRefNet / ISNet) | CPU     |
| [Image Enhancer](./notebooks/image-enhancer)            | 2x / 4x upscaling (Real-ESRGAN) and face restoration (GFPGAN) | CPU, T4 |
| [Speech to Text](./notebooks/speech-to-text)            | Transcripts and .srt / .vtt subtitles from audio or video (faster-whisper) | CPU, T4 |
| [Vocal Remover & Stem Splitter](./notebooks/stem-splitter) | Vocals + instrumental, or four stems, from any song (Demucs) | CPU, T4 |
| [MiniMax-H3](./notebooks/minimax-h3)                    | 5 s text-to-video with matching stereo sound; ~4 min first-run setup, ~3 min per clip (MiniMax H3 Community License) | TPU v5e-1 |

## Adding a notebook

See [CONTRIBUTING.md](./CONTRIBUTING.md). In short: add a folder under
`notebooks/`, run `python scripts/build_index.py`, and open a pull request.
CI rejects a pull request if `index.json` is out of date or a notebook is invalid.

## Using your own collection

Fork this repository and point **NZAP Engine → Settings → Public notebook
collection** at `https://raw.githubusercontent.com/<you>/<fork>/main/`.

## License

Notebooks in this repository are licensed under [Apache-2.0](./LICENSE).
