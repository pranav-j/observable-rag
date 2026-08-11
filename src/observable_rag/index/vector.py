"""Phase 2a: dense vector index over chunks, backed by Qdrant.

Each chunk is embedded with a local sentence-transformers bi-encoder and stored
in Qdrant; search embeds the query and returns the nearest chunks by cosine
similarity. The embedder is injectable so the Qdrant logic can be tested without
loading the model. Point ids are deterministic UUIDs derived from the chunk id,
so re-indexing the same chunk overwrites rather than duplicates.
"""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient, models

from ..ingest.chunk import Chunk

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION = "fastapi_docs"
QDRANT_PATH = "data/index/qdrant"
_NAMESPACE = uuid.UUID("1b671a64-40d5-491e-99b0-da01ff1f3341")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def _load_default_embedder():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)

    def embed(texts):
        return [v.tolist() for v in model.encode(list(texts), normalize_embeddings=True)]

    return embed


class VectorIndex:
    def __init__(self, client: QdrantClient | None = None,
                 collection: str = COLLECTION, embed=None):
        self.client = client or QdrantClient(path=QDRANT_PATH)
        self.collection = collection
        self._embed = embed

    @property
    def embed(self):
        if self._embed is None:
            self._embed = _load_default_embedder()
        return self._embed

    def build(self, chunks: list[Chunk]) -> "VectorIndex":
        vectors = self.embed([c.text for c in chunks])
        dim = len(vectors[0])
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        points = [
            models.PointStruct(
                id=_point_id(c.id),
                vector=vec,
                payload={"chunk_id": c.id, "source": c.source, "url": c.url, "text": c.text},
            )
            for c, vec in zip(chunks, vectors)
        ]
        self.client.upsert(self.collection, points=points)
        return self

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        qv = self.embed([query])[0]
        res = self.client.query_points(self.collection, query=qv, limit=top_k,
                                       with_payload=True)
        return [(p.payload["chunk_id"], float(p.score)) for p in res.points]


_default: "VectorIndex | None" = None


def get_index() -> VectorIndex:
    global _default
    if _default is None:
        _default = VectorIndex()
    return _default


def index_chunks(chunks: list[Chunk]) -> VectorIndex:
    return get_index().build(chunks)


def search(query: str, top_k: int) -> list[tuple[str, float]]:
    return get_index().search(query, top_k)