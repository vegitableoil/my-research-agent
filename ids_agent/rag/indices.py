"""Named RAG index definitions and metadata schemas."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IndexConfig:
    embedding_model: str = "text-embedding-3-large"
    chunk_size: int = 512          # tokens
    chunk_overlap: int = 64
    metadata_schema: dict = field(default_factory=dict)


# Global registry — add new indices here without touching any agent code
INDICES: dict[str, IndexConfig] = {
    "papers_global": IndexConfig(
        chunk_size=512,
        metadata_schema={
            "paper_id": str,
            "title": str,
            "year": int,
            "venue": str,
            "section_type": str,   # abstract | method | results | …
            "topic_tag": str,
        },
    ),
    "code_snippets": IndexConfig(
        chunk_size=256,
        metadata_schema={
            "repo_url": str,
            "language": str,
            "framework": str,
            "model_type": str,
        },
    ),
    "datasets_meta": IndexConfig(
        chunk_size=256,
        metadata_schema={
            "dataset_name": str,
            "num_samples": int,
            "num_features": int,
            "attack_types": list,
        },
    ),
}

# Per-paper indices are created dynamically with the same config as papers_global
PAPER_INDEX_TEMPLATE = IndexConfig(chunk_size=512, chunk_overlap=64)
