"""Subgraph for completing missing details in step plans."""

import asyncio
import json
import logging
import time
from typing import Dict

from ai_prompter import Prompter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from openai import RateLimitError
from typing_extensions import TypedDict

from utils.llm import provision_chat_model
from utils.settings import PipelineSettings
from utils.text import ensure_json_object


class StepCompletionState(TypedDict, total=False):
    document_id: str
    config: PipelineSettings
    outline: Dict[str, object]
    excerpt: str
    tools: list
    step_plan: Dict[str, object]
    completed_plan: Dict[str, object]


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


async def complete_steps(state: StepCompletionState, config: RunnableConfig) -> dict:
    settings = state["config"]
    outline = state.get("outline") or {}
    step_plan = state.get("step_plan") or {}
    tools = state.get("tools") or []
    content = await _invoke_prompt(
        template_name="steps_complete",
        payload_data={
            "document_id": state["document_id"],
            "outline": json.dumps(outline, ensure_ascii=False, indent=2),
            "excerpt": state.get("excerpt", ""),
            "tool_catalog": json.dumps(tools, ensure_ascii=False, indent=2),
            "step_plan": json.dumps(step_plan, ensure_ascii=False, indent=2),
        },
        user_instruction="请输出 JSON。",
        settings=settings,
    )
    plan = ensure_json_object(content)
    if not plan or not isinstance(plan.get("steps"), list):
        raise ValueError("LLM 未能补齐有效的排查步骤 JSON")
    logger.info(
        "complete_steps:done document_id=%s steps=%s",
        state.get("document_id"),
        len(plan.get("steps", [])),
    )
    return {"completed_plan": plan}


workflow = StateGraph(StepCompletionState)
workflow.add_node("complete_steps", complete_steps)
workflow.add_edge(START, "complete_steps")
workflow.add_edge("complete_steps", END)

step_completion_graph = workflow.compile()
