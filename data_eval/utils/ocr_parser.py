"""OCR-backed PDF parsing utilities using a local vLLM OpenAI-compatible endpoint."""

import asyncio
import base64
import io
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import httpx
from pypdf import PdfReader
from pypdf.constants import PagesAttributes as PAGES
from pypdf.generic import IndirectObject

from utils.settings import PipelineSettings


OCR_PROMPT = (
    "提取文档图片中正文的所有信息用 markdown 格式表示，"
    "其中页眉、页脚部分忽略，表格用 html 格式表达，"
    "对于流程图使用Mermaid格式表示，按照阅读顺序组织进行解析。"
)


def _estimate_page_count(pdf_path: Path) -> int:
    with pdf_path.open("rb") as pdf_file:
        reader = PdfReader(pdf_file)
        pages_root = reader.root_object["/Pages"]
        pages_dict = pages_root.get_object() if isinstance(pages_root, IndirectObject) else pages_root
        return int(pages_dict.get(PAGES.COUNT, 0) or 0)


def _render_pdf_pages(
    pdf_path: Path, start_page: int, end_page: int
) -> List[bytes]:
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise RuntimeError("pdf2image is required for OCR rendering") from exc

    rendered: List[bytes] = []
    for page_num in range(start_page, end_page + 1):
        images = convert_from_path(
            str(pdf_path),
            dpi=200,
            first_page=page_num,
            last_page=page_num,
            fmt="png",
        )
        if not images:
            rendered.append(b"")
            continue
        buffer = io.BytesIO()
        images[0].save(buffer, format="PNG")
        rendered.append(buffer.getvalue())
    return rendered


def _image_to_data_url(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


async def _call_ocr_api(
    image_url: str,
    settings: PipelineSettings,
    client: httpx.AsyncClient,
) -> Dict[str, object]:
    if not settings.ocr_api_url:
        raise ValueError("OCR_API_URL is not configured")
    headers = {}
    if settings.ocr_api_key:
        headers["Authorization"] = f"Bearer {settings.ocr_api_key}"
    content = [
        {"type": "text", "text": OCR_PROMPT},
        {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
    ]
    payload = {
        "model": settings.ocr_model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 8000,
    }
    response = await client.post(settings.ocr_api_url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


async def _ocr_pages_parallel(
    image_urls: List[str],
    settings: PipelineSettings,
) -> List[str]:
    timeout = max(5.0, float(settings.ocr_timeout))
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        tasks = [
            _call_ocr_api(image_url, settings, client) for image_url in image_urls
        ]
        results = await asyncio.gather(*tasks)
    texts: List[str] = []
    for result in results:
        choice = (result.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content")
        texts.append(str(text or ""))
    return texts


def iter_pdf_batches(
    pdf_path: str, batch_size: int = 5
) -> Generator[Dict[str, str], None, Dict[str, str]]:
    """
    Yield OCR text batches from a PDF, compatible with utils.pdf_parser.iter_pdf_batches.
    """
    settings = PipelineSettings()
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    total_pages = _estimate_page_count(path)
    metadata: Dict[str, str] = {
        "file_name": path.name,
        "page_count": str(total_pages),
        "title": path.stem,
    }

    start_page = 1
    while start_page <= max(1, total_pages):
        end_page = min(start_page + batch_size - 1, total_pages or start_page + batch_size - 1)
        images = _render_pdf_pages(path, start_page, end_page)
        image_urls = [_image_to_data_url(img) for img in images]
        texts = asyncio.run(_ocr_pages_parallel(image_urls, settings))
        text = "\n\n".join([t.strip() for t in texts if t.strip()])
        yield {
            "start_page": start_page,
            "end_page": end_page,
            "text": text.strip(),
        }
        start_page = end_page + 1

    metadata["page_count"] = str(total_pages or metadata.get("page_count", "0"))
    return metadata


def extract_pdf_text(pdf_path: str) -> Tuple[str, Dict[str, str]]:
    """Backward-compatible helper: read whole PDF into one string via OCR."""
    meta: Dict[str, str] = {}
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
