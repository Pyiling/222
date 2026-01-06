from typing import Dict, List

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from graph.subgraphs import case_builder_graph, ocr_parse_graph
from store.dataset_store import DatasetStore
from utils.settings import PipelineSettings


class PipelineState(TypedDict, total=False):
    pdf_path: str
    document_id: str
    config: PipelineSettings
    metadata: Dict[str, str]
    document_text_path: str
    excerpt: str
    chunk_preview: List[str]
    outline: Dict[str, object]
    trajectory: List[Dict[str, object]]
    case_record: Dict[str, object]


async def run_parse_graph(state: PipelineState, config: RunnableConfig) -> dict:
    parse_state = await ocr_parse_graph.ainvoke(
        {
            "pdf_path": state["pdf_path"],
            "document_id": state["document_id"],
            "config": state["config"],
        }
    )
    return {
        "metadata": parse_state.get("metadata", {}),
        "document_text_path": parse_state.get("document_text_path"),
        "excerpt": parse_state.get("excerpt", ""),
        "chunk_preview": parse_state.get("chunk_preview", []),
    }


async def run_case_builder(state: PipelineState, config: RunnableConfig) -> dict:
    builder_state = await case_builder_graph.ainvoke(
        {
            "document_id": state["document_id"],
            "config": state["config"],
            "metadata": state.get("metadata", {}),
            "document_text_path": state.get("document_text_path"),
            "excerpt": state.get("excerpt", ""),
        }
    )
    return {
        "outline": builder_state.get("outline", {}),
        "trajectory": builder_state.get("trajectory", []),
        "case_record": builder_state.get("case_record", {}),
    }


async def persist_dataset(state: PipelineState, config: RunnableConfig) -> dict:
    settings = state["config"]
    dataset_store = DatasetStore(settings.dataset_store_dir)
    case_record = state.get("case_record")
    if not case_record:
        raise ValueError("case_record not generated")
    dataset_store.append(state["document_id"], [case_record])
    return {}


workflow = StateGraph(PipelineState)
workflow.add_node("run_parse_graph", run_parse_graph)
workflow.add_node("run_case_builder", run_case_builder)
workflow.add_node("persist_dataset", persist_dataset)
workflow.add_edge(START, "run_parse_graph")
workflow.add_edge("run_parse_graph", "run_case_builder")
workflow.add_edge("run_case_builder", "persist_dataset")
workflow.add_edge("persist_dataset", END)

pdf_pipeline_ocr_graph = workflow.compile()
