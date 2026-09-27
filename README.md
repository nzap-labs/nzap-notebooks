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

## Adding a notebook

See [CONTRIBUTING.md](./CONTRIBUTING.md). In short: add a folder under
`notebooks/`, run `python scripts/build_index.py`, and open a pull request.
CI rejects a pull request if `index.json` is out of date or a notebook is invalid.

## Using your own collection

Fork this repository and point **NZAP Engine → Settings → Public notebook
collection** at `https://raw.githubusercontent.com/<you>/<fork>/main/`.

## License

Notebooks in this repository are licensed under [Apache-2.0](./LICENSE).
