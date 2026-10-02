# NZAP apps

An **app** is a notebook with a second file, `app.json`, that tells NZAP Engine
how to present it as a small product instead of as code: a form built from
real widgets, the runtime it needs, how long it takes, and outputs rendered as
an audio player, an image, a table and so on.

```
notebooks/kokoro-tts/
├── notebook.py     the code that runs on the user's Colab runtime
├── notebook.json   metadata and the typed parameters (unchanged)
└── app.json        the app's UI, runtime and timing (this document)
```

A notebook without `app.json` still works everywhere, as a plain notebook. An
app is still a plain notebook too: the Run dialog, the console and the
`.ipynb` export work as before, and the app events below show up as
readable `[nzap] …` lines.

## How an app runs

1. The user picks a runtime, or creates the one `runtime.accelerator` recommends.
2. The engine validates the form values against `notebook.json`, injects
   them as `params`, and runs `notebook.py` on the runtime.
3. The notebook reports progress and results as **app events** (below).
4. The first run on a runtime does the setup (install, download, load).
   The notebook keeps the loaded model in the kernel, so later runs on the
   same runtime skip straight to inference. The app shows this as *warm*.

## `app.json`

```json
{
  "format": "nzap-app/1",
  "icon": "audio-lines",
  "category": "audio",
  "tagline": "Turn text into lifelike speech in seconds.",
  "runtime": { "accelerator": "T4", "supported": ["CPU", "T4", "L4", "A100"], "minVramGb": 2 },
  "estimates": { "setup": 60, "run": 3, "measuredOn": "T4", "runNote": "for about 200 characters" },
  "runLabel": "Generate speech",
  "inputs": [
    { "param": "text", "widget": "textarea", "rows": 6, "maxLength": 5000 },
    { "param": "language", "widget": "segmented", "labels": { "a": "English (US)" } },
    { "param": "voice", "widget": "select", "filter": { "param": "language", "prefix": true } },
    { "param": "speed", "widget": "slider", "min": 0.5, "max": 2, "step": 0.05, "unit": "x" }
  ],
  "outputs": [{ "id": "speech", "kind": "audio", "label": "Speech" }],
  "examples": [{ "label": "Product intro", "values": { "text": "Meet NZAP Engine." } }],
  "links": [{ "label": "Model card", "url": "https://huggingface.co/hexgrad/Kokoro-82M" }],
  "license": "Apache-2.0"
}
```

| Field       | Required | Meaning                                                                                                     |
| ----------- | -------- | ----------------------------------------------------------------------------------------------------------- |
| `format`    | yes      | Always `nzap-app/1`.                                                                                        |
| `category`  | yes      | `audio`, `image`, `video`, `text`, `vision`, `data` or `utility`. Groups apps in the gallery.               |
| `runtime`   | yes      | `accelerator`: the recommended runtime. `supported`: every runtime that works. `highMem`, `minVramGb`: hints. |
| `estimates` | yes      | Seconds on `measuredOn`: `setup` for the first run on a runtime, `run` for a typical warm run.               |
| `outputs`   | yes      | What the app produces (below). At least one.                                                                |
| `inputs`    | no       | Widgets for the parameters, in display order. Parameters left out get a default widget after these.       |
| `icon`      | no       | A [Lucide](https://lucide.dev/icons) icon name. Unknown names fall back to a generic icon.                    |
| `tagline`   | no       | One line for the gallery card (140 characters at most).                                                     |
| `runLabel`  | no       | The run button's text. Default: **Run**.                                                                    |
| `examples`  | no       | Presets that fill the form: `{ "label", "values" }`.                                                        |
| `links`     | no       | `https` links shown on the app page (model card, paper, source).                                            |
| `license`   | no       | The license of the model weights. Say so plainly when they are non-commercial.                              |

Measured times beat guesses: the engine also records how long each app
really took on each accelerator and shows the user's own numbers when it has them.

### Inputs

Each input names a `param` declared in `notebook.json` and a `widget` that
can edit its type:

| Widget                         | Parameter types       | Options                                                         |
| ------------------------------ | --------------------- | --------------------------------------------------------------- |
| `input`                        | `string`              | `placeholder`, `maxLength`                                      |
| `textarea`                     | `string`, `text`      | `rows`, `placeholder`, `maxLength`                              |
| `file`                         | `string`              | `accept`, `maxMb`. Uploads to the runtime; the value is its path. |
| `select`, `segmented`, `radio` | `select`              | `labels` (option → display text), `filter`                      |
| `number`                       | `integer`, `number`   | `min`, `max`, `step`, `unit`                                    |
| `slider`                       | `integer`, `number`   | `min` and `max` (required), `step`, `unit`                      |
| `switch`, `checkbox`           | `boolean`             |                                                                 |

Every widget also takes `label` (overrides the parameter's label) and
`section`: inputs with the same section are grouped under that heading, and a
section named **Advanced** starts collapsed.

`filter: { "param": "language", "prefix": true }` shows only the options that
start with the current value of another parameter. Kokoro uses it so that
picking *English (UK)* (`b`) lists only the `b…` voices.

A `file` input uploads the user's file to
`/content/nzap/inputs/<slug>/<name>` on the runtime before the run and passes
that path as the parameter's value.

## App events

The notebook talks to the app by displaying a bundle with the MIME type
`application/vnd.nzap.app+json`. Add `text/plain` so the console and exports
stay readable:

```python
from IPython.display import display

APP = "kokoro-tts"

def nzap(event, text, **fields):
    payload = {"v": 1, "app": APP, "event": event, **fields}
    display({"application/vnd.nzap.app+json": payload, "text/plain": text}, raw=True)
```

| Event    | Fields                                                    | Shown as                                         |
| -------- | --------------------------------------------------------- | ------------------------------------------------ |
| `stage`  | `id`, `label`, `progress` (0–1, optional)                 | The progress strip: *Installing…*, *Loading…*    |
| `ready`  | `warm` (bool), `setupSeconds`, `device`                   | Marks the end of setup; *warm* skips its estimate |
| `output` | `id`, `kind`, plus the kind's fields (below)              | The result, in the slot declared under `outputs` |
| `done`   | `seconds: { setup, run }`, `warm`                         | The timing, recorded for future estimates        |

Use the stage ids `install`, `download`, `load` and `run` when they fit; the
app gives them their own icons. Errors need no event: raise an exception and
the app shows it.

### Outputs

| `kind`     | Fields                                                              |
| ---------- | ------------------------------------------------------------------- |
| `audio`    | `path` on the runtime, `mime`, `meta` (`duration`, `sampleRate`…)   |
| `image`    | `path`, `mime`, `meta` (`width`, `height`…)                         |
| `video`    | `path`, `mime`                                                      |
| `file`     | `path`, `mime`, `filename`                                          |
| `text`     | `text`                                                              |
| `markdown` | `text`                                                              |
| `json`     | `value`                                                             |
| `table`    | `columns` (strings), `rows` (lists of strings and numbers)          |

Write media to `/content/nzap/outputs/<slug>/` and send its `path`; the app
downloads it from the runtime. That keeps large results out of the
notebook's output and history, and leaves them in **Files** for later.
`text`, `markdown`, `json` and `table` travel inline.

## Writing a good app

- **Cache what is expensive.** Keep models in
  `globals().setdefault("_nzap_apps", {}).setdefault(APP, {})` and check it before
  loading, so the second run is fast.
- **Pin what you download.** Pin package versions, model revisions, and Git
  commits of code you import. Reviewers check that pinned code.
- **Fail with a sentence.** `raise RuntimeError("Breeze needs a GPU runtime (T4 or better).")`
  is the whole error UI.
- **Measure, then fill in `estimates`** from a cold and a warm run on the
  recommended runtime.
- **State the license** of the weights, especially when it limits use.

Validate everything with `python scripts/build_index.py --check`: it checks
`app.json` against this document and against the notebook's parameters.
