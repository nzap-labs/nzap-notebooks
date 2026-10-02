#!/usr/bin/env python3
"""Build and validate the NZAP public notebook catalog.

    python scripts/build_index.py            # rewrite index.json
    python scripts/build_index.py --check    # fail if index.json is stale or invalid (CI)
    python scripts/build_index.py --bundle ../nzap-engine/crates/nzap-core/catalog/bundled.json

Every notebook lives in ``notebooks/<slug>/`` as ``notebook.py`` (the source)
plus ``notebook.json`` (metadata and the parameter schema), and optionally
``app.json``: the UI an NZAP app gets in the engine (see APPS.md). ``index.json``
lists them with the SHA-256 of each source, which NZAP Engine verifies before
running anything it downloads. The rules below mirror the engine's own
validation (``crates/nzap-core/src/notebooks``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
INDEX = ROOT / "index.json"

SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")
KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
TYPES = {"string", "text", "integer", "number", "boolean", "select"}
ALLOWED_META = {"slug", "title", "description", "tags", "author", "params"}
ALLOWED_PARAM = {"key", "label", "type", "default", "required", "options", "description"}
MAX_SOURCE_BYTES = 512 * 1024
MAX_APP_BYTES = 64 * 1024

APP_FORMAT = "nzap-app/1"
APP_FIELDS = {"format", "icon", "category", "tagline", "runtime", "estimates", "runLabel",
              "inputs", "outputs", "examples", "links", "license"}
CATEGORIES = {"audio", "image", "video", "text", "vision", "data", "utility"}
ACCELERATORS = {"CPU", "T4", "L4", "G4", "A100", "H100", "V5E1", "V6E1"}
RUNTIME_FIELDS = {"accelerator", "supported", "highMem", "minVramGb"}
ESTIMATE_FIELDS = {"setup", "run", "measuredOn", "runNote"}
INPUT_FIELDS = {"param", "widget", "label", "placeholder", "rows", "maxLength", "min", "max",
                "step", "unit", "labels", "filter", "accept", "maxMb", "section"}
# Which widgets can edit which parameter types.
WIDGETS = {
    "input": {"string"},
    "textarea": {"string", "text"},
    "file": {"string"},
    "select": {"select"},
    "segmented": {"select"},
    "radio": {"select"},
    "number": {"integer", "number"},
    "slider": {"integer", "number"},
    "switch": {"boolean"},
    "checkbox": {"boolean"},
}
OUTPUT_KINDS = {"audio", "image", "video", "text", "markdown", "json", "table", "file"}
OUTPUT_ID = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


class Invalid(Exception):
    pass


def check_param(slug: str, param: dict, seen: set) -> None:
    where = f"{slug}: parameter {param.get('key')!r}"
    unknown = set(param) - ALLOWED_PARAM
    if unknown:
        raise Invalid(f"{where} has unknown fields {sorted(unknown)}")
    if not isinstance(param.get("key"), str) or not KEY.match(param["key"]):
        raise Invalid(f"{where}: key must be a Python identifier")
    if param["key"] in seen:
        raise Invalid(f"{where}: declared twice")
    seen.add(param["key"])
    if not isinstance(param.get("label"), str) or not param["label"].strip():
        raise Invalid(f"{where}: a label is required")
    if param.get("type") not in TYPES:
        raise Invalid(f"{where}: type must be one of {sorted(TYPES)}")
    if param["type"] == "select":
        options = param.get("options")
        if not isinstance(options, list) or not options or not all(isinstance(o, str) for o in options):
            raise Invalid(f"{where}: select parameters need a non-empty list of string options")
    default = param.get("default")
    if default is not None and not isinstance(default, (str, int, float, bool)):
        raise Invalid(f"{where}: default must be a string, number or boolean")
    if param["type"] == "select" and default is not None and default not in param["options"]:
        raise Invalid(f"{where}: default must be one of the options")
    if "required" in param and not isinstance(param["required"], bool):
        raise Invalid(f"{where}: required must be true or false")


def number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def check_app(slug: str, app: dict, params: list) -> None:
    """Validate app.json against the notebook's parameters (APPS.md)."""
    where = f"{slug}/app.json"

    def need(condition, message):
        if not condition:
            raise Invalid(f"{where}: {message}")

    need(isinstance(app, dict), "must be an object")
    unknown = set(app) - APP_FIELDS
    need(not unknown, f"unknown fields {sorted(unknown)}")
    need(app.get("format") == APP_FORMAT, f"format must be {APP_FORMAT!r}")
    need(app.get("category") in CATEGORIES, f"category must be one of {sorted(CATEGORIES)}")
    for field in ("icon", "tagline", "runLabel", "license"):
        need(field not in app or isinstance(app[field], str), f"{field} must be a string")
    need(len(app.get("tagline", "")) <= 140, "tagline is longer than 140 characters")

    runtime = app.get("runtime")
    need(isinstance(runtime, dict) and not set(runtime) - RUNTIME_FIELDS, "runtime must be an object")
    need(runtime.get("accelerator") in ACCELERATORS, f"runtime.accelerator must be one of {sorted(ACCELERATORS)}")
    supported = runtime.get("supported", [runtime["accelerator"]])
    need(isinstance(supported, list) and set(supported) <= ACCELERATORS, "runtime.supported lists unknown accelerators")
    need(runtime["accelerator"] in supported, "runtime.supported must include the recommended accelerator")
    need("highMem" not in runtime or isinstance(runtime["highMem"], bool), "runtime.highMem must be true or false")
    need("minVramGb" not in runtime or number(runtime["minVramGb"]), "runtime.minVramGb must be a number")

    estimates = app.get("estimates")
    need(isinstance(estimates, dict) and not set(estimates) - ESTIMATE_FIELDS, "estimates must be an object")
    for field in ("setup", "run"):
        need(number(estimates.get(field)) and estimates[field] >= 0, f"estimates.{field} must be seconds")

    declared = {param["key"]: param for param in params}
    seen = set()
    inputs = app.get("inputs", [])
    need(isinstance(inputs, list), "inputs must be a list")
    for item in inputs:
        need(isinstance(item, dict) and not set(item) - INPUT_FIELDS, f"input {item!r} has unknown fields")
        key = item.get("param")
        need(key in declared, f"input {key!r} is not a declared parameter")
        need(key not in seen, f"input {key!r} is listed twice")
        seen.add(key)
        kind = declared[key]["type"]
        widget = item.get("widget")
        need(widget in WIDGETS, f"input {key!r}: widget must be one of {sorted(WIDGETS)}")
        need(kind in WIDGETS[widget], f"input {key!r}: a {widget} cannot edit a {kind} parameter")
        if widget == "slider":
            need(number(item.get("min")) and number(item.get("max")) and item["min"] < item["max"],
                 f"input {key!r}: a slider needs numeric min < max")
        labels = item.get("labels", {})
        need(isinstance(labels, dict) and all(isinstance(v, str) for v in labels.values()),
             f"input {key!r}: labels must map values to strings")
        if labels and kind == "select":
            need(set(labels) <= set(declared[key]["options"]), f"input {key!r}: labels name unknown options")
        rule = item.get("filter")
        if rule is not None:
            need(kind == "select", f"input {key!r}: only select inputs can be filtered")
            need(isinstance(rule, dict) and rule.get("param") in declared and rule["param"] != key,
                 f"input {key!r}: filter.param must name another parameter")
            need(rule.get("prefix") is True, f"input {key!r}: filter.prefix must be true")
        need("section" not in item or isinstance(item["section"], str), f"input {key!r}: section must be a string")

    outputs = app.get("outputs")
    need(isinstance(outputs, list) and outputs, "declare at least one output")
    ids = set()
    for output in outputs:
        need(isinstance(output, dict) and OUTPUT_ID.match(str(output.get("id", ""))), f"output {output!r} needs an id")
        need(output["id"] not in ids, f"output {output['id']!r} is declared twice")
        ids.add(output["id"])
        need(output.get("kind") in OUTPUT_KINDS, f"output {output['id']!r}: kind must be one of {sorted(OUTPUT_KINDS)}")

    for example in app.get("examples", []):
        need(isinstance(example, dict) and isinstance(example.get("label"), str), "examples need a label")
        values = example.get("values", {})
        need(isinstance(values, dict) and set(values) <= set(declared), f"example {example['label']!r} sets unknown parameters")
        for key, value in values.items():
            if declared[key]["type"] == "select":
                need(value in declared[key]["options"], f"example {example['label']!r}: {key} is not an option")

    for link in app.get("links", []):
        need(isinstance(link, dict) and isinstance(link.get("label"), str)
             and str(link.get("url", "")).startswith("https://"), "links need a label and an https url")


def load(folder: Path) -> tuple[dict, str]:
    slug = folder.name
    meta_path, source_path = folder / "notebook.json", folder / "notebook.py"
    if not meta_path.is_file() or not source_path.is_file():
        raise Invalid(f"{slug}: needs notebook.json and notebook.py")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    unknown = set(meta) - ALLOWED_META
    if unknown:
        raise Invalid(f"{slug}: unknown fields {sorted(unknown)}")
    if meta.get("slug") != slug or not SLUG.match(slug):
        raise Invalid(f"{slug}: slug must match the folder name and ^[a-z0-9][a-z0-9-]{{1,62}}$")
    if not isinstance(meta.get("title"), str) or not meta["title"].strip():
        raise Invalid(f"{slug}: a title is required")
    if len(meta.get("description", "")) > 2000:
        raise Invalid(f"{slug}: description is longer than 2000 characters")
    seen: set = set()
    for param in meta.get("params", []):
        check_param(slug, param, seen)
    raw = source_path.read_bytes()
    if len(raw) > MAX_SOURCE_BYTES:
        raise Invalid(f"{slug}: notebook.py is larger than {MAX_SOURCE_BYTES} bytes")
    source = raw.decode("utf-8")
    compile(source, f"{slug}/notebook.py", "exec")  # syntax check only; never executed
    app_path = folder / "app.json"
    if app_path.is_file():
        raw_app = app_path.read_bytes()
        if len(raw_app) > MAX_APP_BYTES:
            raise Invalid(f"{slug}: app.json is larger than {MAX_APP_BYTES} bytes")
        meta["app"] = json.loads(raw_app.decode("utf-8"))
        check_app(slug, meta["app"], meta.get("params", []))
    return meta, source


def build() -> tuple[dict, dict]:
    entries, sources = [], {}
    for folder in sorted(p for p in NOTEBOOKS.iterdir() if p.is_dir()):
        meta, source = load(folder)
        entries.append(
            {
                "slug": meta["slug"],
                "title": meta["title"].strip(),
                "description": meta.get("description", ""),
                "tags": meta.get("tags", []),
                "author": meta.get("author"),
                "params": meta.get("params", []),
                "source": f"notebooks/{meta['slug']}/notebook.py",
                "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                **({"app": meta["app"]} if "app" in meta else {}),
            }
        )
        sources[meta["slug"]] = source
    entries.sort(key=lambda entry: entry["title"].lower())
    return {"version": 1, "notebooks": entries}, sources


def dump(value: dict) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="validate without writing")
    parser.add_argument("--bundle", type=Path, help="also write an engine snapshot with inline sources")
    args = parser.parse_args()
    try:
        index, sources = build()
    except (Invalid, SyntaxError, ValueError) as error:
        print(f"invalid catalog: {error}", file=sys.stderr)
        return 1

    text = dump(index)
    if args.check:
        current = INDEX.read_text(encoding="utf-8") if INDEX.exists() else ""
        if current != text:
            print("index.json is out of date: run python scripts/build_index.py", file=sys.stderr)
            return 1
        print(f"catalog ok: {len(index['notebooks'])} notebooks")
    else:
        INDEX.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {INDEX.relative_to(ROOT)} ({len(index['notebooks'])} notebooks)")

    if args.bundle:
        bundle = {
            "version": 1,
            "notebooks": [dict(entry, sourceText=sources[entry["slug"]]) for entry in index["notebooks"]],
        }
        args.bundle.parent.mkdir(parents=True, exist_ok=True)
        args.bundle.write_text(dump(bundle), encoding="utf-8", newline="\n")
        print(f"wrote {args.bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
