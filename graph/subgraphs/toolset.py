"""Toolset extraction subgraph."""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional

from ai_prompter import Prompter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from openai import RateLimitError
from typing_extensions import TypedDict

from utils.llm import provision_chat_model
from utils.settings import PipelineSettings
from utils.text import ensure_json_list


class ToolsetState(TypedDict, total=False):
    document_id: str
    config: PipelineSettings
    outline: Dict[str, object]
    tools: List[Dict[str, object]]


logger = logging.getLogger(__name__)

#尝试读取本地固定的工具清单。
def _load_fixed_toolset(settings: PipelineSettings) -> Optional[List[Dict[str, object]]]:
    toolset_name = settings.toolset_name
    if not toolset_name:
        return None
    toolset_path = settings.tools_dir / toolset_name
    if not toolset_path.exists():
        raise FileNotFoundError(f"Toolset not found: {toolset_path}")
    content = toolset_path.read_text(encoding="utf-8")
    tools = json.loads(content)
    if not isinstance(tools, list):
        raise ValueError("Toolset must be a JSON array")
    return [tool for tool in tools if isinstance(tool, dict)]

#把读取到的工具或者 LLM 输出的工具列表标准化。
def _normalize_toolset(tools: List[Dict[str, object]]) -> List[Dict[str, object]]:
    normalized: List[Dict[str, object]] = []
    for tool in tools:
        if "function" in tool:
            function = tool.get("function", {})
            name = function.get("name")
            if not name:
                continue
            tool.setdefault("type", "function")
            function.setdefault("description", "")
            function.setdefault("parameters", {"type": "object", "properties": {}, "required": []})
            function.setdefault("return", {"type": "string", "description": ""})
            normalized.append(tool)
            continue

        name = tool.get("name")
        if not name:
            continue
        description = tool.get("description", "")
        args_schema = tool.get("args_schema", {})
        properties: Dict[str, object] = {}
        required: List[str] = []
        if isinstance(args_schema, dict):
            for arg_name, arg_meta in args_schema.items():
                required.append(arg_name)
                if isinstance(arg_meta, dict):
                    arg_type = arg_meta.get("type", "string")
                    arg_desc = arg_meta.get("description", "")
                else:
                    arg_type = "string"
                    arg_desc = str(arg_meta)
                properties[arg_name] = {"type": arg_type, "description": arg_desc}
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
                "return": {"type": "string", "description": ""},
            },
        }
        normalized.append(schema)
    if not normalized:
        raise ValueError("工具清单为空或缺少 name 字段")
    return normalized

#当固定工具不存在时，用 LLM 生成工具清单。
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

#提取工具调用思维链。
async def extract_toolset(state: ToolsetState, config: RunnableConfig) -> dict:
    settings = state["config"]
    outline = state.get("outline") or {}
    tools = _load_fixed_toolset(settings)
    if tools is None:
        content = await _invoke_prompt(
            template_name="toolset_plan",
            payload_data={
                "document_id": state["document_id"],
                "case_blueprint": json.dumps(outline, ensure_ascii=False, indent=2),
            },
            user_instruction="请输出 JSON 数组。",
            settings=settings,
        )
        tools = ensure_json_list(content)
    tools = _normalize_toolset(tools)
    logger.info(
        "extract_toolset:done document_id=%s tools=%s source=%s",
        state.get("document_id"),
        len(tools),
        "fixed" if settings.toolset_name else "llm",
    )
    return {"tools": tools}


workflow = StateGraph(ToolsetState)
workflow.add_node("extract_toolset", extract_toolset)
workflow.add_edge(START, "extract_toolset")
workflow.add_edge("extract_toolset", END)

toolset_graph = workflow.compile()
