import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass
class PipelineSettings:
    """Central place for environment-driven settings."""

    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_api_base: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    openai_context_window: int = field(
        default_factory=lambda: int(os.getenv("OPENAI_CONTEXT_WINDOW", "120000"))
    )
    openai_timeout: float = field(
        default_factory=lambda: float(os.getenv("OPENAI_TIMEOUT", "30"))
    )
    document_store_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DOCUMENT_STORE_DIR", "data/documents"))
    )
    dataset_store_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATASET_STORE_DIR", "data/datasets"))
    )
    dataset_store_eval: Path = field(
        default_factory=lambda: Path(os.getenv("DATASET_STORE_EVAL", "data/datasets_evaluation"))
    )
    cache_dir: Path = field(
        default_factory=lambda: Path(os.getenv("CACHE_DIR", "data/cache"))
    )
    pdf_batch_size: int = field(
        default_factory=lambda: int(os.getenv("PDF_BATCH_SIZE", "1"))
    )
    tools_dir: Path = field(
        default_factory=lambda: Path(os.getenv("TOOLS_DIR", "data/tools"))
    )
    toolset_name: str | None = field(
        default_factory=lambda: os.getenv("TOOLSET_NAME") or None
    )
    # toolset_name: str | None = field(
    #     default_factory=lambda: r"C:\Users\pan\Desktop\Data_Analyze_LiAgent-main\data_pipeline\data\tools\all_tools.json"
    # )
    ocr_api_url: str = field(default_factory=lambda: os.getenv("OCR_API_URL", ""))
    ocr_api_key: str = field(default_factory=lambda: os.getenv("OCR_API_KEY", ""))
    ocr_model: str = field(
        default_factory=lambda: os.getenv("OCR_MODEL", "") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    ocr_timeout: float = field(
        default_factory=lambda: float(os.getenv("OCR_TIMEOUT", "60"))
    )

    def validate(self) -> None:
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured")
