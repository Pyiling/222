import argparse
import asyncio
from pathlib import Path
import uuid
import logging
from graph.pipeline_eval import evaluation_pipeline
from utils.settings import PipelineSettings
import os
print(repr(os.getenv("OPENAI_API_KEY")))
print(repr(os.getenv("OPENAI_BASE_URL")))
print(repr(os.getenv("OPENAI_MODEL")))


import asyncio
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PDF to dataset pipeline")
    parser.add_argument(
        "pdf_path",
        nargs="?",
        default=r"C:\Users\pan\Desktop\Data_Analyze_LiAgent-main\data_pipeline\data\documents\sample_0.pdf",  # 这里设置默认路径
        help="Path to the PDF file to process",
    )
    parser.add_argument(
        "--document-id",
        dest="document_id",
        help="Optional identifier for outputs/cache. Defaults to PDF 文件名（无扩展名）。",
    )
    parser.add_argument(
        "--parse-mode",
        choices=("pdf", "ocr"),
        default="pdf",
        help="Choose parsing mode: pdf (default) or ocr.",
    )
    return parser.parse_args()


async def run_pipeline_eval(pdf_path: str, document_id: str, parse_mode: str) -> None:
    settings = PipelineSettings()
    settings.validate()
    graph = evaluation_pipeline
    await graph.ainvoke(
        {
            "pdf_path": pdf_path,
            "document_id": document_id,
            "config": settings,
        }
    )


def main() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    default_id = Path(args.pdf_path).stem
    document_id = args.document_id or default_id or uuid.uuid4().hex
    asyncio.run(run_pipeline_eval(args.pdf_path, document_id, args.parse_mode))


if __name__ == "__main__":
    main()

