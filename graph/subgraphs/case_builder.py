"""Subgraph responsible for producing structured SFT records."""

import asyncio
import json
import logging
import operator
from pathlib import Path
import time
from typing import Annotated, Dict, List

from ai_prompter import Prompter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from openai import RateLimitError
from typing_extensions import TypedDict

from utils.llm import provision_chat_model
from utils.settings import PipelineSettings
from graph.subgraphs.step_completion import step_completion_graph
from graph.subgraphs.step_extraction import step_extraction_graph
from graph.subgraphs.toolset import toolset_graph
from graph.subgraphs.trajectory_builder import trajectory_builder_graph
from utils.text import chunk_text, ensure_json_object


class CaseBuilderState(TypedDict, total=False):
    document_id: str
    config: PipelineSettings
    metadata: Dict[str, str]
    document_text_path: str
    excerpt: str
    context_chunks: Annotated[List[str], operator.add]
    context_excerpt: str
    outline: Dict[str, object]
    step_plan: Dict[str, object]
    completed_plan: Dict[str, object]
    trajectory: Annotated[List[Dict[str, object]], operator.add]
    case_record: Dict[str, object]


logger = logging.getLogger(__name__)


async def prepare_context(state: CaseBuilderState, config: RunnableConfig) -> dict:
    start = time.perf_counter()
    settings = state["config"]
    document_text_path = state.get("document_text_path")
    if not document_text_path:
        raise ValueError("document_text_path missing in CaseBuilder state")
    text_path = Path(document_text_path)
    document_text = text_path.read_text(encoding="utf-8")
    context_chunks = chunk_text(document_text, chunk_size=1800, overlap=200, max_chunks=4)
    fallback_excerpt = state.get("excerpt") or ""
    excerpt_body = "\n\n".join(context_chunks) or fallback_excerpt or document_text[:2000]
    excerpt = fallback_excerpt or excerpt_body
    approx_char_budget = max(2000, settings.openai_context_window * 4)
    context_excerpt = excerpt_body[:approx_char_budget]
    return {
        "context_chunks": context_chunks,
        "context_excerpt": context_excerpt,
        "excerpt": excerpt,
    }


async def _invoke_prompt(
    template_name: str,
    payload_data: Dict[str, object],
    user_instruction: str,
    settings: PipelineSettings,
) -> str:
    start = time.perf_counter()
    prompt = Prompter(prompt_template=template_name)
    prompt_text = prompt.render(payload_data)
    messages = [SystemMessage(content=prompt_text), HumanMessage(content=user_instruction)]
    attempts = 3
    for attempt in range(1, attempts + 1):
        model = await provision_chat_model(prompt_text, settings)
        try:
            response = await model.ainvoke(messages)
            content = response.content if isinstance(response.content, str) else json.dumps(response.content)
            logger.info(
                "llm_call:done template=%s duration=%.2fs",
                template_name,
                time.perf_counter() - start,
            )
            return content
        except RateLimitError:
            if attempt == attempts:
                raise
            await asyncio.sleep(2 * attempt)
    raise RuntimeError("LLM 调用失败")


async def draft_case_outline(state: CaseBuilderState, config: RunnableConfig) -> dict:
    settings = state["config"]
    content = await _invoke_prompt(
        template_name="case_outline",
        payload_data={
            "document_id": state["document_id"],
            "metadata": json.dumps(state.get("metadata", {}), ensure_ascii=False, indent=2),
            "excerpt": state.get("context_excerpt") or state.get("excerpt", ""),
        },
        user_instruction="请输出 JSON。",
        settings=settings,
    )
    outline = ensure_json_object(content)
    if not outline:
        raise ValueError("LLM 未能生成案例规划 JSON")
    outline.setdefault("language", "zh")
    outline.setdefault("id", state["document_id"])
    outline.setdefault("domain", outline.get("domain") or "general_incident")
    outline.setdefault("meta", {})
    outline.setdefault("problem", {})
    outline.setdefault("labels", {})
    logger.info(
        "draft_case_outline:done document_id=%s",
        state.get("document_id"),
    )
    return {"outline": outline}


async def run_toolset_graph(state: CaseBuilderState, config: RunnableConfig) -> dict:
    start = time.perf_counter()
    outline = state.get("outline") or {}
    toolset_state = await toolset_graph.ainvoke(
        {
            "document_id": state["document_id"],
            "config": state["config"],
            "outline": outline,
        }
    )
    tools = toolset_state.get("tools", [])
    outline["tools"] = tools
    logger.info(
        "run_toolset_graph:done document_id=%s tools=%s duration=%.2fs",
        state.get("document_id"),
        len(tools),
        time.perf_counter() - start,
    )
    return {"outline": outline}


async def run_step_extraction(state: CaseBuilderState, config: RunnableConfig) -> dict:
    outline = state.get("outline") or {}
    step_state = await step_extraction_graph.ainvoke(
        {
            "document_id": state["document_id"],
            "config": state["config"],
            "outline": outline,
            "excerpt": state.get("context_excerpt") or state.get("excerpt", ""),
            "tools": outline.get("tools", []),
        }
    )
    return {"step_plan": step_state.get("step_plan", {})}


async def run_step_completion(state: CaseBuilderState, config: RunnableConfig) -> dict:
    outline = state.get("outline") or {}
    step_state = await step_completion_graph.ainvoke(
        {
            "document_id": state["document_id"],
            "config": state["config"],
            "outline": outline,
            "excerpt": state.get("context_excerpt") or state.get("excerpt", ""),
            "tools": outline.get("tools", []),
            "step_plan": state.get("step_plan", {}),
        }
    )
    return {"completed_plan": step_state.get("completed_plan", {})}


async def run_trajectory_builder(state: CaseBuilderState, config: RunnableConfig) -> dict:
    outline = state.get("outline") or {}
    trajectory_state = await trajectory_builder_graph.ainvoke(
        {
            "document_id": state["document_id"],
            "config": state["config"],
            "outline": outline,
            "tools": outline.get("tools", []),
            "completed_plan": state.get("completed_plan", {}),
        }
    )
    return {"trajectory": trajectory_state.get("trajectory", [])}


async def finalize_case_record(state: CaseBuilderState, config: RunnableConfig) -> dict:
    outline = state.get("outline") or {}
    trajectory = state.get("trajectory") or []
    case_record = {
        "id": outline.get("id") or state.get("document_id"),
        "domain": outline.get("domain", "general_incident"),
        "language": outline.get("language", "zh"),
        "meta": outline.get("meta", {}),
        "problem": outline.get("problem", {}),
        "tools": outline.get("tools", []),
        "trajectory": trajectory,
        "labels": outline.get("labels", {}),
    }
    logger.info(
        "finalize_case_record:done document_id=%s",
        state.get("document_id"),
    )
    return {"case_record": case_record}


workflow = StateGraph(CaseBuilderState)
workflow.add_node("prepare_context", prepare_context)
workflow.add_node("draft_case_outline", draft_case_outline)
workflow.add_node("run_toolset_graph", run_toolset_graph)
workflow.add_node("run_step_extraction", run_step_extraction)
workflow.add_node("run_step_completion", run_step_completion)
workflow.add_node("run_trajectory_builder", run_trajectory_builder)
workflow.add_node("finalize_case_record", finalize_case_record)
workflow.add_edge(START, "prepare_context")
workflow.add_edge("prepare_context", "draft_case_outline")
workflow.add_edge("draft_case_outline", "run_toolset_graph")
workflow.add_edge("run_toolset_graph", "run_step_extraction")
workflow.add_edge("run_step_extraction", "run_step_completion")
workflow.add_edge("run_step_completion", "run_trajectory_builder")
workflow.add_edge("run_trajectory_builder", "finalize_case_record")
workflow.add_edge("finalize_case_record", END)

case_builder_graph = workflow.compile()