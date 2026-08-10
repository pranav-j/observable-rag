"""Phase 1: fetch and parse the FastAPI docs (Markdown) into Documents.

Reads English docs under docs/en/docs/**.md straight from the FastAPI repo. We
download the repo archive once from codeload (no API rate limits, reproducible)
and read the Markdown out of it, so each file yields a stable slug = its path
under that folder, e.g. "tutorial/response-status-code". Non-content meta files
(underscore-prefixed, e.g. _llm-test) are skipped. Pin `ref` to a tag or commit
SHA for full reproducibility; it defaults to the repo's default branch.
"""

from __future__ import annotations

import io
import tarfile
from dataclasses import dataclass

import httpx

REPO = "fastapi/fastapi"
DOCS_SEGMENT = "/docs/en/docs/"
DEFAULT_REF = "master"
SITE_BASE = "https://fastapi.tiangolo.com/"
_CODELOAD = "https://codeload.github.com"


@dataclass
class Document:
    source: str  # stable slug, e.g. "tutorial/response-status-code"
    url: str      # human-facing docs URL (best effort)
    text: str     # raw Markdown


def _relpath(name: str) -> str:
    return name.split(DOCS_SEGMENT, 1)[1]


def _is_content(name: str) -> bool:
    # skip non-content meta files, e.g. docs/en/docs/_llm-test.md
    return not any(part.startswith("_") for part in _relpath(name).split("/"))


def _site_url(slug: str) -> str:
    if slug == "index":
        path = ""
    elif slug.endswith("/index"):
        path = slug[: -len("index")]
    else:
        path = slug + "/"
    return SITE_BASE + path


def fetch_corpus(ref: str = DEFAULT_REF, limit: int | None = None) -> list[Document]:
    """Fetch English FastAPI docs as Documents. `limit` caps the count (dev/CI)."""
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        resp = client.get(f"{_CODELOAD}/{REPO}/tar.gz/{ref}")
        resp.raise_for_status()
        archive = io.BytesIO(resp.content)

    docs: list[Document] = []
    with tarfile.open(fileobj=archive, mode="r:gz") as tar:
        members = sorted(
            (m for m in tar.getmembers()
             if m.isfile() and DOCS_SEGMENT in m.name
             and m.name.endswith(".md") and _is_content(m.name)),
            key=lambda m: m.name,
        )
        if limit:
            members = members[:limit]
        for m in members:
            fh = tar.extractfile(m)
            if fh is None:
                continue
            text = fh.read().decode("utf-8", errors="replace")
            slug = _relpath(m.name)[: -len(".md")]
            docs.append(Document(source=slug, url=_site_url(slug), text=text))
    return docs
