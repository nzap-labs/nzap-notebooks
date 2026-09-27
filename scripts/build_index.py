#!/usr/bin/env python3
"""Build and validate the NZAP public notebook catalog.

    python scripts/build_index.py            # rewrite index.json
    python scripts/build_index.py --check    # fail if index.json is stale or invalid (CI)
    python scripts/build_index.py --bundle ../nzap-engine/crates/nzap-core/catalog/bundled.json

Every notebook lives in ``notebooks/<slug>/`` as ``notebook.py`` (the source)
plus ``notebook.json`` (metadata and the parameter schema). ``index.json``
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
