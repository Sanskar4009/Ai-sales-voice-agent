"""RAG service for indexing and retrieving Fortivex product knowledge."""

import re
import threading
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_KB_PATH = Path(__file__).resolve().parents[2] / "data" / "product_kb.md"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class ProductKnowledgeRAG:
    """Builds and queries a FAISS vector index over the product knowledge base."""

    def __init__(
        self,
        kb_path: Path = DEFAULT_KB_PATH,
        model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        self.kb_path = Path(kb_path)
        self.model_name = model_name
        self._lock = threading.Lock()
        self._model: Optional[SentenceTransformer] = None
        self._index: Optional[faiss.IndexFlatIP] = None
        self._chunks: list[str] = []

    @staticmethod
    def split_markdown_semantically(markdown_text: str, max_chars: int = 1100) -> list[str]:
        """Split markdown into heading-aware semantic chunks."""
        lines = markdown_text.splitlines()
        sections: list[tuple[str, str, list[str]]] = []
        h1_title = ""
        h2_title = ""
        h3_title = ""
        current_lines: list[str] = []

        def flush_section() -> None:
            body = "\n".join(current_lines).strip()
            # Skip empty blocks or horizontal-rule-only blocks
            cleaned = re.sub(r"^-{3,}$", "", body, flags=re.MULTILINE).strip()
            if not cleaned:
                return
            header_parts = [p for p in (h1_title, h2_title, h3_title) if p]
            heading_prefix = " > ".join(header_parts)
            sections.append((heading_prefix, cleaned, header_parts))

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# ") and not stripped.startswith("## "):
                flush_section()
                current_lines = []
                h1_title = stripped[2:].strip()
                h2_title = ""
                h3_title = ""
            elif stripped.startswith("## ") and not stripped.startswith("### "):
                flush_section()
                current_lines = []
                h2_title = stripped[3:].strip()
                h3_title = ""
            elif stripped.startswith("### "):
                flush_section()
                current_lines = []
                h3_title = stripped[4:].strip()
            elif stripped == "---":
                continue
            else:
                current_lines.append(line)

        flush_section()

        chunks: list[str] = []
        for heading_prefix, body, _ in sections:
            full_chunk = f"[{heading_prefix}]\n{body}" if heading_prefix else body
            if len(full_chunk) <= max_chars:
                chunks.append(full_chunk)
                continue

            # Split oversized sections by paragraph/block boundaries while preserving heading
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
            buffer = ""
            for para in paragraphs:
                candidate = f"{buffer}\n\n{para}".strip() if buffer else para
                formatted = f"[{heading_prefix}]\n{candidate}" if heading_prefix else candidate
                if len(formatted) <= max_chars:
                    buffer = candidate
                else:
                    if buffer:
                        chunks.append(
                            f"[{heading_prefix}]\n{buffer}" if heading_prefix else buffer
                        )
                    buffer = para
            if buffer:
                chunks.append(
                    f"[{heading_prefix}]\n{buffer}" if heading_prefix else buffer
                )

        return chunks

    def _ensure_initialized(self) -> None:
        """Load embedding model, chunk knowledge base, and build FAISS index once."""
        if self._index is not None and self._model is not None:
            return

        with self._lock:
            if self._index is not None and self._model is not None:
                return

            if not self.kb_path.exists():
                raise FileNotFoundError(f"Knowledge base file not found: {self.kb_path}")

            kb_text = self.kb_path.read_text(encoding="utf-8")
            self._chunks = self.split_markdown_semantically(kb_text)
            if not self._chunks:
                raise ValueError(f"No semantic chunks extracted from {self.kb_path}")

            try:
                self._model = SentenceTransformer(
                    self.model_name, local_files_only=True
                )
            except Exception:
                self._model = SentenceTransformer(self.model_name)
            embeddings = self._model.encode(
                self._chunks,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            vectors = np.asarray(embeddings, dtype=np.float32)

            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            self._index = index

    def initialize(self) -> int:
        """Eagerly initialize the FAISS index and return the number of indexed chunks."""
        self._ensure_initialized()
        return len(self._chunks)

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """Return the top_k most relevant knowledge base chunks for a user query."""
        cleaned_query = query.strip()
        if not cleaned_query:
            return ""

        self._ensure_initialized()
        assert self._model is not None
        assert self._index is not None

        k = max(1, min(top_k, len(self._chunks)))
        query_embedding = self._model.encode(
            [cleaned_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        query_vector = np.asarray(query_embedding, dtype=np.float32)
        _, indices = self._index.search(query_vector, k)

        matched_chunks = [
            self._chunks[idx]
            for idx in indices[0]
            if 0 <= idx < len(self._chunks)
        ]
        return "\n\n---\n\n".join(matched_chunks)


_rag_service = ProductKnowledgeRAG()


def init_rag() -> int:
    """Initialize the singleton RAG index at startup."""
    return _rag_service.initialize()


def retrieve_context(query: str, top_k: int = 3) -> str:
    """Retrieve the most relevant Fortivex product knowledge chunks for a query."""
    return _rag_service.retrieve_context(query=query, top_k=top_k)
