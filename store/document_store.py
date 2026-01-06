import json
from pathlib import Path
from typing import Dict


class DocumentStore:
    """Extremely simple filesystem-backed document registry."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, document_id: str, text_input: str | Path, metadata: Dict[str, str]) -> Path:
        if isinstance(text_input, Path):
            text = text_input.read_text(encoding="utf-8")
        else:
            text = text_input
        payload = {
            "document_id": document_id,
            "metadata": metadata,
            "text": text,
        }
        target = self.base_dir / f"{document_id}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target
