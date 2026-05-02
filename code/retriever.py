"""Domain-specific TF-IDF retriever."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from code.config import DOMAIN_TO_CORPUS, RETRIEVAL_TOP_K
from code.models import RetrievedDoc


def _chunk_text(text: str, max_words: int = 220) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks: list[str] = []
    for start in range(0, len(words), max_words):
        chunks.append(" ".join(words[start : start + max_words]))
    return chunks


def _read_docs_from_folder(folder: Path) -> list[tuple[str, str]]:
    """Load docs recursively and split large files into retrieval chunks."""
    docs: list[tuple[str, str]] = []
    if not folder.exists():
        return docs
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()

        if suffix in {".txt", ".md"}:
            content = path.read_text(encoding="utf-8").replace("\ufeff", "").strip()
            if content:
                for idx, chunk in enumerate(_chunk_text(content)):
                    docs.append((f"{path}#chunk-{idx+1}", chunk))
        elif suffix == ".json":
            raw = path.read_text(encoding="utf-8").replace("\ufeff", "").strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, list):
                for item_idx, item in enumerate(payload, start=1):
                    if isinstance(item, dict):
                        text = str(item.get("content") or item.get("text") or item.get("body") or "").strip()
                    else:
                        text = str(item).strip()
                    if text:
                        for chunk_idx, chunk in enumerate(_chunk_text(text), start=1):
                            docs.append((f"{path}#item-{item_idx}-chunk-{chunk_idx}", chunk))
    return docs


@dataclass
class DomainIndex:
    sources: list[str]
    texts: list[str]
    vectorizer: TfidfVectorizer | None
    matrix: object | None


class DomainRetriever:
    """Retrieves top-k docs from a predicted domain only.

    Indexes are built once at startup for better scalability and stable latency.
    """

    def __init__(self) -> None:
        self.indices: dict[str, DomainIndex] = {}
        for domain, folder in DOMAIN_TO_CORPUS.items():
            docs = _read_docs_from_folder(folder)
            sources = [source for source, _ in docs]
            texts = [text for _, text in docs]
            if not texts:
                self.indices[domain] = DomainIndex(
                    sources=sources,
                    texts=texts,
                    vectorizer=None,
                    matrix=None,
                )
                continue

            vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
            matrix = vectorizer.fit_transform(texts)
            self.indices[domain] = DomainIndex(
                sources=sources,
                texts=texts,
                vectorizer=vectorizer,
                matrix=matrix,
            )

    def retrieve(self, domain: str, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[RetrievedDoc]:
        if domain not in self.indices:
            return []
        index = self.indices[domain]
        if not index.texts or index.vectorizer is None or index.matrix is None:
            return []
        query_vector = index.vectorizer.transform([query])
        sims = cosine_similarity(query_vector, index.matrix)[0]
        scored = sorted(enumerate(sims), key=lambda item: item[1], reverse=True)[:top_k]

        return [
            RetrievedDoc(source=index.sources[idx], content=index.texts[idx], score=float(score))
            for idx, score in scored
            if score > 0
        ]
