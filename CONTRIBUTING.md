# Contributing a notebook

Thanks for sharing. Every notebook here runs on other people's Colab runtimes
with their Google credentials, so reviews are strict.

## Checklist

1. Create `notebooks/<slug>/` with a slug of lowercase letters, digits and
   dashes (2–63 characters).
2. Add `notebook.py`. It is plain Python, reads its inputs from `params`, and
   prints its results.
3. Add `notebook.json` with `slug`, `title`, `description`, `tags`, `author`
   and `params` (see the README).
4. Run `python scripts/build_index.py` and commit the updated `index.json`.
5. Open a pull request describing what the notebook does and which runtime it
   needs (CPU / GPU / TPU).

## Review rules

A notebook is not accepted if it:

- sends data anywhere other than the service it clearly exists to use (for
  example Hugging Face for a Hugging Face model);
- asks for or reads credentials, tokens or `google.colab.auth` without an
  obvious, documented reason;
- downloads and executes code at runtime (`curl | sh`, `exec(requests.get(...))`);
- mines cryptocurrency or otherwise breaks the
  [Colab terms](https://research.google.com/colaboratory/faq.html);
- is obfuscated, or larger than 512 KB.

Installing dependencies with `pip` is fine when the notebook needs them.

## Local check

```bash
python scripts/build_index.py --check
```

The same command runs in CI on every pull request.
