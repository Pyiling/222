import json
from typing import Any, Dict, Iterable, List, Optional


def chunk_text(
    text: str, chunk_size: int = 2000, overlap: int = 200, max_chunks: Optional[int] = None
) -> List[str]:
    """Split text into overlapping chunks for LLM consumption."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    clean_text = text.strip()
    if not clean_text:
        return []

    chunks = []
    start = 0
    length = len(clean_text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(clean_text[start:end])
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
        if end == length:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def clean_json_output(message: str) -> str:
    """Remove code fences or stray commentary from model output."""
    content = message.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
    if content.endswith("```"):
        content = content.rsplit("\n", 1)[0]
    if content.lower().startswith("json"):
        content = content[4:].lstrip()
    return content.strip()


def ensure_json_list(payload: str) -> List[dict]:
    """Parse JSON safely, ensuring we always return a list of dicts."""
    cleaned = clean_json_output(payload)
    if not cleaned:
        return []
    data = json.loads(cleaned)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, Iterable):
        return [item for item in data if isinstance(item, dict)]
    return []


def ensure_json_object(payload: str) -> Dict[str, Any]:
    """Parse JSON safely, returning the first dict-like structure."""
    cleaned = clean_json_output(payload)
    if not cleaned:
        return {}
    data = json.loads(cleaned)
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                return item
    return {}
