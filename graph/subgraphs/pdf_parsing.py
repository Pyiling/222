"""Document parsing and persistence subgraph."""

import json
import logging
import operator
from pathlib import Path
import time
from typing import Annotated, Dict, List

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from store.cache_store import CacheStore
from store.document_store import DocumentStore
from utils.pdf_parser import iter_pdf_batches
from utils.settings import PipelineSettings
from utils.text import chunk_text


class ParseState(TypedDict, total=False):
    pdf_path: str
    document_id: str
    config: PipelineSettings
    metadata: Dict[str, str]
    document_text_path: str
    excerpt: str
    chunk_preview: Annotated[List[str], operator.add]


logger = logging.getLogger(__name__)


def _ensure_document_id(value: str | None) -> str:
    if value:
        return value
    raise ValueError("document_id must be provided")


def _load_cached_parse(
    settings: PipelineSettings,
    document_id: str,
) -> Dict[str, object] | None:
    doc_cache_dir = settings.cache_dir / document_id
    text_path = doc_cache_dir / "full_text.txt"
    if not text_path.exists():
        return None
    batch_files = sorted(doc_cache_dir.glob("batch_*.json"))
    if not batch_files:
        return None

    document_text = text_path.read_text(encoding="utf-8")
    chunk_preview = chunk_text(document_text, max_chunks=5)
    excerpt_limit = min(settings.openai_context_window * 4, 200_000)
    excerpt = document_text[:excerpt_limit].strip()

    metadata: Dict[str, str] = {"batches": str(len(batch_files))}
    document_path = settings.document_store_dir / f"{document_id}.json"
    if document_path.exists():
        try:
            payload = json.loads(document_path.read_text(encoding="utf-8"))
            doc_meta = payload.get("metadata")
            if isinstance(doc_meta, dict):
                metadata.update({str(k): str(v) for k, v in doc_meta.items()})
        except json.JSONDecodeError:
            pass

    return {
        "document_text_path": str(text_path),
        "metadata": metadata,
        "chunk_preview": chunk_preview,
        "excerpt": excerpt,
    }

#核心解析 PDF → 文本批次 → 缓存
async def parse_pdf(state: ParseState, config: RunnableConfig) -> dict:
    start = time.perf_counter()
    settings = state["config"]
    cache_store = CacheStore(settings.cache_dir)
    document_id = _ensure_document_id(state.get("document_id"))

    cached = _load_cached_parse(settings, document_id)
    if cached:
        logger.info(
            "parse_pdf:cache_hit document_id=%s duration=%.2fs",
            document_id,
            time.perf_counter() - start,
        )
        return cached

    chunk_preview: List[str] = []
    metadata: Dict[str, str] = {}
    batch_index = 0
    excerpt_parts: List[str] = []
    excerpt_limit = min(settings.openai_context_window * 4, 200_000)
    chunk_preview_limit = 5

    doc_cache_dir = settings.cache_dir / document_id
    doc_cache_dir.mkdir(parents=True, exist_ok=True)
    text_path = doc_cache_dir / "full_text.txt"

    gen = iter_pdf_batches(state["pdf_path"], batch_size=settings.pdf_batch_size)
    with text_path.open("w", encoding="utf-8") as text_file:
        try:
            while True:
                batch = next(gen)
                if text_file.tell() > 0:
                    text_file.write("\n\n")
                text_file.write(batch["text"])
                cache_store.save_batch(
                    document_id=document_id,
                    batch_index=batch_index,
                    start_page=batch["start_page"],
                    end_page=batch["end_page"],
                    text=batch["text"],
                    metadata=metadata or {},
                )
                if len(chunk_preview) < chunk_preview_limit:
                    remaining = chunk_preview_limit - len(chunk_preview)
                    chunk_preview.extend(chunk_text(batch["text"], max_chunks=remaining))
                if excerpt_limit > 0 and batch["text"]:
                    excerpt_slice = batch["text"][:excerpt_limit]
                    excerpt_parts.append(excerpt_slice)
                    excerpt_limit -= len(excerpt_slice)
                batch_index += 1
        except StopIteration as stop:
            metadata = stop.value or {}

    excerpt_text = "\n\n".join(part.strip() for part in excerpt_parts if part).strip()
    metadata = metadata or {}
    metadata.update({"batches": str(batch_index)})
    logger.info(
        "parse_pdf:done document_id=%s batches=%s duration=%.2fs",
        document_id,
        batch_index,
        time.perf_counter() - start,
    )
    return {
        "document_text_path": str(text_path),
        "metadata": metadata,
        "chunk_preview": chunk_preview,
        "excerpt": excerpt_text,
    }

#将解析后的 PDF 文本和 metadata 存储到 DocumentStore
async def persist_document(state: ParseState, config: RunnableConfig) -> dict:
    start = time.perf_counter()
    settings = state["config"]
    document_store = DocumentStore(settings.document_store_dir)
    document_id = _ensure_document_id(state.get("document_id"))
    document_text_path = state.get("document_text_path")
    if not document_text_path:
        raise ValueError("document_text_path missing from state")
    document_store.save(document_id, Path(document_text_path), state.get("metadata", {}))
    logger.info(
        "persist_document:done document_id=%s duration=%.2fs",
        document_id,
        time.perf_counter() - start,
    )
    return {
        "document_id": document_id,
        "document_text_path": document_text_path,
    }


workflow = StateGraph(ParseState)
workflow.add_node("parse_pdf", parse_pdf)
workflow.add_node("persist_document", persist_document)
workflow.add_edge(START, "parse_pdf")
workflow.add_edge("parse_pdf", "persist_document")
workflow.add_edge("persist_document", END)

parse_graph = workflow.compile()
