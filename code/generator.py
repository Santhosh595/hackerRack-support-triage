"""Grounded response generator using retrieved snippets only."""

from __future__ import annotations

from pathlib import Path

from code.models import RetrievedDoc


def _short_source(source: str) -> str:
    """Return just the filename (and chunk id) from a full path source string."""
    # source may be like /abs/path/to/file.txt#chunk-1
    if "#" in source:
        path_part, chunk_part = source.rsplit("#", 1)
        return f"{Path(path_part).name}#{chunk_part}"
    return Path(source).name


def generate_grounded_response(docs: list[RetrievedDoc]) -> str:
    if not docs:
        return ""
    # Keep generation extractive and deterministic to avoid unsupported claims.
    lines = ["Based on our documentation:"]
    for idx, doc in enumerate(docs, start=1):
        snippet = " ".join(doc.content.split()[:45]).strip()
        lines.append(f"{idx}. {snippet} (source: {_short_source(doc.source)})")
    return "\n".join(lines)


def build_justification(docs: list[RetrievedDoc], stage_reasons: list[str]) -> str:
    avg_score = sum(doc.score for doc in docs) / len(docs) if docs else 0.0
    reasons = "; ".join(stage_reasons)
    return f"{reasons}; retrieval_docs={len(docs)}; avg_similarity={avg_score:.3f}"
