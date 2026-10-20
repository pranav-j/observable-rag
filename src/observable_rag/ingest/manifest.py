"""Corpus integrity manifest: the golden set's half of the chunk-id contract.

Chunk ids are stable by construction (f"{source}#{index}"), but that only pins
WHERE a chunk sits, not WHAT it says. The FastAPI docs move: a paragraph added
near the top of a page re-packs every chunk below it, so `tutorial/foo#3` keeps
its id while its text becomes something the golden set was never verified
against. Retrieval metrics stay green and quietly measure the wrong thing.

So we record a hash of the text behind every chunk id the golden set cites, and
fail loudly when one changes. Only cited chunks are hashed: those are the ones a
human actually checked, and a whole-corpus hash would break on every unrelated
docs edit.

    python -m observable_rag.ingest.manifest          # write the manifest
    python -m observable_rag.ingest.manifest --check  # verify, exit 1 on drift
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

CHUNKS_PATH = Path("data/corpus/chunks.jsonl")
GOLDEN_PATH = Path("data/eval/golden_set.jsonl")
MANIFEST_PATH = Path("data/eval/corpus_manifest.json")
MOVING_REFS = {"master", "main", "HEAD"}


def _warn_if_unpinned(ref: str) -> None:
    if ref in MOVING_REFS:
        print(f"  ! '{ref}' is a moving branch, not a pin: these hashes describe the docs "
              f"as of this build and will drift. Set FASTAPI_REF to a tag or commit sha.")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_chunk_texts(path: Path = CHUNKS_PATH) -> dict[str, str]:
    texts: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            texts[d["id"]] = d["text"]
    return texts


def golden_chunk_ids(path: Path = GOLDEN_PATH) -> list[str]:
    """Every chunk id cited by the golden set, deduplicated, in a stable order."""
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.update(json.loads(line).get("relevant_chunk_ids", []))
    return sorted(ids)


def build_manifest(chunks: dict[str, str], cited: list[str], ref: str) -> dict:
    missing = [c for c in cited if c not in chunks]
    if missing:
        raise KeyError(f"golden set cites chunk ids absent from the corpus: {missing}")
    return {
        "fastapi_ref": ref,
        "n_chunks": len(chunks),
        "n_cited": len(cited),
        "cited_chunks": {c: text_hash(chunks[c]) for c in cited},
    }


def check_manifest(manifest: dict, chunks: dict[str, str],
                   ids_only: bool = False) -> list[str]:
    """Return a list of human-readable problems; empty means the contract holds.

    ids_only checks existence but not content. A missing id breaks the eval
    outright and is always a hard failure; drifting text is a softer signal that
    depends on whether the docs revision is pinned yet, so CI reports the two
    separately.
    """
    problems: list[str] = []
    for chunk_id, expected in manifest["cited_chunks"].items():
        if chunk_id not in chunks:
            problems.append(f"{chunk_id}: cited by the golden set but missing from the corpus")
        elif not ids_only and text_hash(chunks[chunk_id]) != expected:
            problems.append(
                f"{chunk_id}: text changed since the golden set was verified "
                f"({expected} -> {text_hash(chunks[chunk_id])})")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="verify the corpus against the manifest instead of writing it")
    ap.add_argument("--ids-only", action="store_true",
                    help="with --check: require the cited ids to EXIST, but don't "
                         "fail on text drift")
    ap.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    ap.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    ap.add_argument("--ref", default=None,
                    help="the docs revision these chunks were built from")
    args = ap.parse_args()

    chunks = load_chunk_texts(args.chunks)

    if args.check:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        problems = check_manifest(manifest, chunks, ids_only=args.ids_only)
        for p in problems:
            print(f"  ! {p}")
        if problems:
            print(f"\n{len(problems)} chunk(s) drifted. The golden set was verified against "
                  f"{manifest['fastapi_ref']}; re-verify the affected items or rebuild "
                  f"from that revision.")
            return 1
        scope = "exist" if args.ids_only else "match the manifest"
        print(f"ok: {manifest['n_cited']} cited chunks {scope} "
              f"({manifest['fastapi_ref']})")
        _warn_if_unpinned(manifest["fastapi_ref"])
        return 0

    from .load import DEFAULT_REF  # local: keeps --check from needing httpx
    ref = args.ref or os.getenv("FASTAPI_REF", DEFAULT_REF)
    cited = golden_chunk_ids(args.golden)
    manifest = build_manifest(chunks, cited, ref)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(cited)} cited-chunk hashes -> {args.manifest} (ref: {ref})")
    _warn_if_unpinned(ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
