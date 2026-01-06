from pathlib import Path
from typing import Dict, Generator, Iterator, Tuple

from pypdf import PdfReader
from pypdf._page import PageObject
from pypdf.constants import PageAttributes as PG
from pypdf.constants import PagesAttributes as PAGES
from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject, NameObject


INHERITABLE_PAGE_ATTRS = (
    NameObject(PG.RESOURCES),
    NameObject(PG.MEDIABOX),
    NameObject(PG.CROPBOX),
    NameObject(PG.ROTATE),
)


def _iter_pages(reader: PdfReader) -> Iterator[PageObject]:
    root_pages_obj = reader.root_object["/Pages"]
    if isinstance(root_pages_obj, DictionaryObject):
        root_pages = root_pages_obj
    else:
        root_pages = root_pages_obj.get_object()
    stack: list[tuple[DictionaryObject, Dict[NameObject, object], IndirectObject | None]] = [
        (root_pages, {}, None)
    ]

    while stack:
        node, inherit, reference = stack.pop()
        node_type = node.get("/Type")
        if node_type is None and PAGES.KIDS not in node:
            node_type = "/Page"

        if node_type == "/Pages":
            next_inherit = inherit.copy()
            for attr in INHERITABLE_PAGE_ATTRS:
                if attr in node:
                    next_inherit[attr] = node[attr]
            kids_obj = node.get(PAGES.KIDS, ArrayObject())
            if isinstance(kids_obj, IndirectObject):
                kids_obj = kids_obj.get_object()
            if not isinstance(kids_obj, ArrayObject):
                continue
            for kid in reversed(kids_obj):
                kid_ref = kid if isinstance(kid, IndirectObject) else None
                kid_obj = kid.get_object()
                if isinstance(kid_obj, DictionaryObject):
                    stack.append((kid_obj, next_inherit.copy(), kid_ref))
        elif node_type == "/Page":
            page_dict = DictionaryObject()
            page_dict.update(node)
            for attr, value in inherit.items():
                page_dict.setdefault(attr, value)
            page = PageObject(reader, reference)
            page.update(page_dict)
            yield page


def iter_pdf_batches(
    pdf_path: str, batch_size: int = 5
) -> Generator[Dict[str, str], None, Dict[str, str]]:
    """
    Yield page batches from a PDF to keep memory usage low.

    Each yield returns a dict containing:
        - start_page (int, 1-based)
        - end_page   (int, 1-based, inclusive)
        - text       (str)

    The final return value (StopIteration) carries the metadata dict.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with path.open("rb") as pdf_file:
        reader = PdfReader(pdf_file)
        pages_root = reader.root_object["/Pages"]
        pages_dict = pages_root.get_object() if isinstance(pages_root, IndirectObject) else pages_root
        total_pages = int(pages_dict.get(PAGES.COUNT, 0) or 0)
        metadata = {
            "file_name": path.name,
            "page_count": str(total_pages),
            "title": getattr(reader.metadata, "title", None) or path.stem,
        }

        buffer = []
        start_page = 1
        last_page_num = 0
        for idx, page in enumerate(_iter_pages(reader), start=1):
            last_page_num = idx
            text = page.extract_text() or ""
            buffer.append(text.strip())
            if len(buffer) >= batch_size:
                batch_text = "\n\n".join(buffer).strip()
                yield {
                    "start_page": start_page,
                    "end_page": idx,
                    "text": batch_text,
                }
                buffer = []
                start_page = idx + 1
        if buffer:
            batch_text = "\n\n".join(buffer).strip()
            yield {
                "start_page": start_page,
                "end_page": last_page_num,
                "text": batch_text,
            }

    metadata["page_count"] = str(total_pages or last_page_num)
    return metadata


def extract_pdf_text(pdf_path: str) -> Tuple[str, Dict[str, str]]:
    """Backward-compatible helper: read whole PDF into one string."""
    meta = {}
    chunks = []
    gen = iter_pdf_batches(pdf_path, batch_size=5)
    try:
        while True:
            batch = next(gen)
            chunks.append(batch["text"])
    except StopIteration as stop:
        meta = stop.value or {}
    document_text = "\n\n".join(chunks).strip()
    return document_text, meta
