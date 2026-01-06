"""Subgraph for turning step plans into assistant/tool trajectories."""

import asyncio
import json
import logging
import time
from typing import Dict, List

from ai_prompter import Prompter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from openai import RateLimitError
from typing_extensions import TypedDict

from utils.llm import provision_chat_model
from utils.settings import PipelineSettings
from utils.text import ensure_json_object


class TrajectoryState(TypedDict, total=False):
    document_id: str
    config: PipelineSettings
    outline: Dict[str, object]
    tools: list
    completed_plan: Dict[str, object]
    trajectory: List[Dict[str, object]]


logger = logging.getLogger(__name__)


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


def _assign_turn_id(turns: List[Dict[str, object]], start_id: int) -> int:
    current = start_id
    for turn in turns:
        turn["turn_id"] = current
        current += 1
    return current


async def build_trajectory(state: TrajectoryState, config: RunnableConfig) -> dict:
    settings = state["config"]
    outline = state.get("outline") or {}
    tools = state.get("tools") or []
    completed_plan = state.get("completed_plan") or {}
    steps = completed_plan.get("steps") or []
    if not steps:
        raise ValueError("completed_plan.steps 为空，无法生成轨迹")

    case_summary = json.dumps(outline, ensure_ascii=False, indent=2)
    history: List[Dict[str, object]] = []
    trajectory: List[Dict[str, object]] = []
    next_id = 1

    for idx, step in enumerate(steps):
        content = await _invoke_prompt(
            template_name="trajectory_step",
            payload_data={
                "case_summary": case_summary,
                "plan_step": json.dumps(step, ensure_ascii=False, indent=2),
                "history": json.dumps(history[-4:], ensure_ascii=False, indent=2),
                "tool_catalog": json.dumps(tools, ensure_ascii=False, indent=2),
                "is_first_step": idx == 0,
            },
            user_instruction="请输出 JSON。",
            settings=settings,
        )
        step_payload = ensure_json_object(content)
        assistant_turn = step_payload.get("assistant_turn")
        tool_turn = step_payload.get("tool_turn")
        if not assistant_turn or not tool_turn:
            raise ValueError("轨迹步骤 JSON 缺少 assistant_turn 或 tool_turn")
        next_id = _assign_turn_id([assistant_turn, tool_turn], next_id)
        trajectory.extend([assistant_turn, tool_turn])
        history.extend([assistant_turn, tool_turn])

    final_requirements = completed_plan.get("final_requirements", {})
    content = await _invoke_prompt(
        template_name="trajectory_final",
        payload_data={
            "case_summary": case_summary,
            "history": json.dumps(history, ensure_ascii=False, indent=2),
            "final_requirements": json.dumps(final_requirements, ensure_ascii=False, indent=2),
        },
        user_instruction="请输出 JSON。",
        settings=settings,
    )
    final_payload = ensure_json_object(content)
    final_turn = final_payload.get("final_turn")
    if not final_turn:
        raise ValueError("最终总结 JSON 缺少 final_turn")
    _assign_turn_id([final_turn], next_id)
    trajectory.append(final_turn)

    logger.info(
        "build_trajectory:done document_id=%s turns=%s",
        state.get("document_id"),
        len(trajectory),
    )
    return {"trajectory": trajectory}


workflow = StateGraph(TrajectoryState)
workflow.add_node("build_trajectory", build_trajectory)
workflow.add_edge(START, "build_trajectory")
workflow.add_edge("build_trajectory", END)

trajectory_builder_graph = workflow.compile()
