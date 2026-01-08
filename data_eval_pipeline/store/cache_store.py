import json
from pathlib import Path
from typing import Dict

#于缓存解析后的 PDF 批次数据，可以避免每次都重新解析 PDF
class CacheStore:
    """Filesystem cache for parsed PDF batches."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _doc_dir(self, document_id: str) -> Path:
        doc_dir = self.base_dir / document_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        return doc_dir

    def save_batch(
        self,
        document_id: str,
        batch_index: int,
        start_page: int,
        end_page: int,
        text: str,
        metadata: Dict[str, str],
    ) -> Path:
        doc_dir = self._doc_dir(document_id)
        target = doc_dir / f"batch_{batch_index:04d}.json"
        payload = {
            "document_id": document_id,
            "batch_index": batch_index,
            "start_page": start_page,
            "end_page": end_page,
            "text": text,
            "metadata": metadata,
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target
