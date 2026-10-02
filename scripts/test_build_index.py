#!/usr/bin/env python3
"""Checks that build_index.py rejects malformed app.json files (APPS.md)."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_index  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PARAMS = [
    {"key": "text", "label": "Text", "type": "text"},
    {"key": "voice", "label": "Voice", "type": "select", "options": ["af_a", "bf_b"]},
    {"key": "lang", "label": "Language", "type": "select", "options": ["a", "b"]},
    {"key": "speed", "label": "Speed", "type": "number"},
    {"key": "flag", "label": "Flag", "type": "boolean"},
]
GOOD = {
    "format": "nzap-app/1",
    "category": "audio",
    "runtime": {"accelerator": "T4", "supported": ["CPU", "T4"]},
    "estimates": {"setup": 10, "run": 1},
    "inputs": [
        {"param": "text", "widget": "textarea"},
        {"param": "voice", "widget": "select", "filter": {"param": "lang", "prefix": True}},
        {"param": "speed", "widget": "slider", "min": 0.5, "max": 2},
        {"param": "flag", "widget": "switch", "section": "Advanced"},
    ],
    "outputs": [{"id": "speech", "kind": "audio"}],
    "examples": [{"label": "One", "values": {"voice": "af_a"}}],
    "links": [{"label": "Card", "url": "https://example.com"}],
}


def mutate(path: str, value) -> dict:
    app = copy.deepcopy(GOOD)
    target = app
    keys = path.split(".")
    for key in keys[:-1]:
        target = target[int(key)] if key.isdigit() else target[key]
    last = keys[-1]
    if value is KeyError:
        del target[last]
    elif last.isdigit():
        target[int(last)] = value
    else:
        target[last] = value
    return app


BAD = {
    "wrong format": mutate("format", "nzap-app/2"),
    "unknown field": mutate("extra", 1),
    "unknown category": mutate("category", "music"),
    "no outputs": mutate("outputs", []),
    "bad output kind": mutate("outputs.0.kind", "hologram"),
    "unknown accelerator": mutate("runtime.accelerator", "V100"),
    "recommended not supported": mutate("runtime.supported", ["CPU"]),
    "missing setup estimate": mutate("estimates.setup", KeyError),
    "undeclared param": mutate("inputs.0.param", "nope"),
    "widget type mismatch": mutate("inputs.0.widget", "slider"),
    "slider without range": mutate("inputs.2", {"param": "speed", "widget": "slider"}),
    "filter on itself": mutate("inputs.1.filter", {"param": "voice", "prefix": True}),
    "example unknown option": mutate("examples.0.values", {"voice": "zz"}),
    "http link": mutate("links.0.url", "http://example.com"),
}


def main() -> int:
    build_index.check_app("good", GOOD, PARAMS)
    failures = []
    for name, app in BAD.items():
        try:
            build_index.check_app("bad", app, PARAMS)
        except build_index.Invalid:
            continue
        except (KeyError, TypeError, AttributeError) as error:
            failures.append(f"{name}: crashed with {type(error).__name__} instead of Invalid")
            continue
        failures.append(f"{name}: accepted")
    # Every app in the repository passes too.
    for folder in sorted((ROOT / "notebooks").iterdir()):
        if (folder / "app.json").is_file():
            meta = json.loads((folder / "notebook.json").read_text(encoding="utf-8"))
            build_index.check_app(folder.name, json.loads((folder / "app.json").read_text(encoding="utf-8")), meta["params"])
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"app.json validator ok: rejected {len(BAD)} malformed apps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
