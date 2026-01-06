import logging
import time
from typing import Dict, List

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from graph.evaluation.cot_evaluator import evaluation_graph
from graph.subgraphs import case_builder_graph, parse_graph
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
    evaluation_results: Dict[str, object]  # 新增：评估结果

logger = logging.getLogger(__name__)

# 解析 PDF → 抽取文本、生成摘要/分块
async def run_parse_graph(state: PipelineState, config: RunnableConfig) -> dict:
    start = time.perf_counter()
    logger.info("parse_graph:start document_id=%s", state.get("document_id"))
    parse_state = await parse_graph.ainvoke(
        {
            "pdf_path": state["pdf_path"],
            "document_id": state["document_id"],
            "config": state["config"],
        }
    )
    logger.info(
        "parse_graph:done document_id=%s duration=%.2fs",
        state.get("document_id"),
        time.perf_counter() - start,
    )
    return {
        "metadata": parse_state.get("metadata", {}),
        "document_text_path": parse_state.get("document_text_path"),
        "excerpt": parse_state.get("excerpt", ""),
        "chunk_preview": parse_state.get("chunk_preview", []),
    }

#构建案例 → LLM 或规则生成案例大纲、分析路径、完整记录
async def run_case_builder(state: PipelineState, config: RunnableConfig) -> dict:
    start = time.perf_counter()
    logger.info("case_builder:start document_id=%s", state.get("document_id"))
    builder_state = await case_builder_graph.ainvoke(
        {
            "document_id": state["document_id"],
            "config": state["config"],
            "metadata": state.get("metadata", {}),
            "document_text_path": state.get("document_text_path"),
            "excerpt": state.get("excerpt", ""),
        }
    )
    logger.info(
        "case_builder:done document_id=%s duration=%.2fs",
        state.get("document_id"),
        time.perf_counter() - start,
    )
    return {
        "outline": builder_state.get("outline", {}),
        "trajectory": builder_state.get("trajectory", []),
        "case_record": builder_state.get("case_record", {}),
    }


# 评估案例 → 对生成的案例进行质量评估
async def run_evaluation(state: PipelineState, config: RunnableConfig) -> dict:
    start = time.perf_counter()
    logger.info("evaluation:start document_id=%s", state.get("document_id"))

    # 获取案例记录进行评估
    case_record = state.get("case_record", {})

    # 调用评估子图
    eval_state = await evaluation_graph.ainvoke(
        {
            "document_id": state["document_id"],
            "config": state["config"],
            "case_record": case_record,
            "metadata": state.get("metadata", {}),
            "excerpt": state.get("excerpt", ""),
            "trajectory": state.get("trajectory", []),
        }
    )
    print("eval_state keys:", eval_state.keys())

    logger.info(
        "evaluation:done document_id=%s duration=%.2fs",
        state.get("document_id"),
        time.perf_counter() - start,
    )

    return {
        "evaluation_results": eval_state.get("evaluation_results", {}),
    }
# 根据评分决定走向
async def check_evaluation_score(state: PipelineState, config: RunnableConfig) -> str:
    """
    如果 overall_score >= 0.9 → 保存
    否则 → 重新生成思维链
    """
    evaluation_results = state.get("evaluation_results", {})
    overall_score = evaluation_results.get("overall_score", 0.0)

    logger.info(
        "check_evaluation_score: document_id=%s overall_score=%.3f",
        state.get("document_id"),
        overall_score,
    )

    if overall_score >= 0.9:
        return "persist"
    else:
        return "regenerate"

#持久化数据 → 保存到本地或数据库，形成可复用的数据集
async def persist_dataset(state: PipelineState, config: RunnableConfig) -> dict:
    start = time.perf_counter()
    settings = state["config"]
    dataset_store = DatasetStore(settings.dataset_store_dir)
    case_record = state.get("case_record")
    if not case_record:
        raise ValueError("case_record not generated")
    dataset_store.append(state["document_id"], [case_record])
    dataset_eval_store = DatasetStore(settings.dataset_store_eval)
    eval_record = state.get("evaluation_results")
    if not eval_record:
        raise ValueError("eval_record not generated")
    dataset_eval_store.append(state["document_id"], [eval_record])
    logger.info(
        "persist_dataset:done document_id=%s duration=%.2fs",
        state.get("document_id"),
        time.perf_counter() - start,
    )
    return {}


# 创建并配置工作流
workflow = StateGraph(PipelineState)
workflow.add_node("run_parse_graph", run_parse_graph)
workflow.add_node("run_case_builder", run_case_builder)
workflow.add_node("run_evaluation", run_evaluation)  # 新增评估节点
workflow.add_node("check_evaluation_score", check_evaluation_score)
workflow.add_node("persist_dataset", persist_dataset)

# 配置工作流连接
workflow.add_edge(START, "run_parse_graph")
workflow.add_edge("run_parse_graph", "run_case_builder")
workflow.add_edge("run_case_builder", "run_evaluation")  # 案例构建完成后进行评估
workflow.add_conditional_edges(
    "run_evaluation",
    check_evaluation_score,
    {
        "persist": "persist_dataset",        # 分数 ≥ 0.9
        "regenerate": "run_case_builder",    # 分数 < 0.9 → 重新生成
    }
)
workflow.add_edge("persist_dataset", END)

pdf_pipeline_graph = workflow.compile()
