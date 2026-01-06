import json
from pathlib import Path
from typing import Iterable, Mapping


class DatasetStore:
    """Append-only JSONL store for LLM-ready training records."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def append(self, document_id: str, records: Iterable[Mapping]) -> Path:
        target = self.base_dir / f"{document_id}.jsonl"
        with target.open("w", encoding="utf-8") as fh:
            for record in records:
                serialized = json.dumps(record, ensure_ascii=False, indent=2)
                fh.write(serialized + "\n")
        return target
