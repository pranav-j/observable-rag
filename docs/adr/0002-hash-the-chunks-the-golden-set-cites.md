# 2. Hash the chunks the golden set cites

Date: 2026-10-20

## Context

Chunk ids are the contract between retrieval and evaluation. A golden item says
"the answer to this question is in `tutorial/bigger-applications#2`", and every
retrieval metric is computed against ids like that one.

The ids are stable by construction — `f"{source}#{index}"`, index from zero —
but that only fixes *where* a chunk sits in a page, not *what it says*. Chunks
are packed greedily to a token budget, so a paragraph added near the top of a
page re-packs everything below it. The id survives; the text behind it changes.

This is not hypothetical. The corpus the current golden set was verified against
was built on an unpinned `master`. Rebuilding from `master` roughly six weeks
later gave:

- 953 chunks before, 972 after
- 1 id gone, 20 new
- **531 of 952 surviving ids holding different text**
- one of those, `tutorial/bigger-applications#2`, cited by the golden set

Of the 531, 500 are `release-notes`, which is a changelog that grows from the
top — every one of its chunks shifts on every release. The other 31 are ordinary
documentation edits.

An id-existence check would have caught the one deleted id and none of the rest.
Every retrieval metric would have stayed green while measuring against text no
human ever verified.

## Decision

Record a short SHA-256 of the text behind every chunk id the golden set cites,
in `data/eval/corpus_manifest.json`, alongside the docs revision it was built
from. `python -m observable_rag.ingest.manifest` writes it; `--check` verifies
the corpus against it and exits non-zero on drift.

Two levels, because the two failures differ in severity:

- `--check --ids-only` — a cited id has disappeared. The eval is broken. Hard
  failure in CI.
- `--check` — the text behind a cited id changed. The labels may no longer
  describe what's there. Reported in CI as a warning, because it is expected to
  fire constantly until the docs revision is pinned.

Only cited chunks are hashed. A whole-corpus hash would break on every unrelated
documentation edit and teach everyone to ignore it.

## Alternatives

- **Commit the corpus.** Removes the drift entirely, and costs a few megabytes
  of generated data in git plus a merge conflict on every rebuild. Worth
  reconsidering if the corpus stops being cheap to rebuild.
- **Content-addressed chunk ids** (hash instead of position). Ids would stop
  lying, but every id would change whenever its text changed, breaking the
  golden set loudly and often, and they'd no longer be readable in a trace.
- **Hash the whole corpus.** One number, no idea which chunk moved, fires on
  every unrelated edit.
- **Trust the pin alone.** Pinning the revision is the actual fix for drift, but
  a pin is a setting someone can change; the manifest is the check that notices.

## Consequences

- The manifest must be refreshed whenever the golden set gains items or the
  corpus is deliberately rebuilt: `python -m observable_rag.ingest.manifest`.
  `tests/test_golden_integrity.py` fails if the manifest and the golden set fall
  out of step, so forgetting is caught.
- The manifest currently records `fastapi_ref: master`, which is honest and
  useless as a pin. Both the builder and the checker print a warning saying so.
  Pinning the revision is a separate decision with a migration attached, since
  it changes the corpus and may invalidate verified labels.
- Drift is reported, not repaired. When a cited chunk's text changes, a human
  has to re-read that chunk and decide whether the label still holds.
