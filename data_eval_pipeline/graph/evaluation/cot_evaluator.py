"""Subgraph responsible for evaluating case quality with detailed tool usage metrics."""

import asyncio
import json
import logging
import operator
import time
from typing import Annotated, Dict, List, Any
from enum import Enum

from ai_prompter import Prompter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from openai import RateLimitError
from typing_extensions import TypedDict

from utils.llm import provision_chat_model
from utils.settings import PipelineSettings


class EvaluationState(TypedDict, total=False):
    document_id: str
    config: PipelineSettings
    case_record: Dict[str, object]
    excerpt: str
    trajectory: List[Dict[str, object]]

    # 评估结果字段
    tool_executability_results: Annotated[List[Dict], operator.add]
    tool_workflow_results: Dict[str, object]
    tool_necessity_results: Annotated[List[Dict], operator.add]
    tool_causality_results: Annotated[List[Dict], operator.add]
    tool_causal_chain_results: Annotated[List[Dict], operator.add]  # 新增：工具链因果连贯性
    correctness_results: Dict[str, object]
    evaluation_results: Dict[str, object]
    factual_consistency_results: Dict[str, object]
    supplementary_rationality_results: Dict[str, object]

    tool_executability_score: float
    tool_necessity_score: float
    tool_causality_score: float
    tool_causal_chain_score: float
    factual_consistency_score: float
    supplementary_rationality_score: float


logger = logging.getLogger(__name__)


class ToolEvaluationMetric(Enum):
    """工具评估指标枚举"""
    EXECUTABILITY = "tool_executability"  # 工具可执行性（规则）
    WORKFLOW = "tool_workflow"  # 工具调用流程合理性（规则）
    NECESSITY = "tool_necessity"  # 工具使用必要性（LLM）
    CAUSALITY = "tool_causality"  # 工具贡献性（LLM）
    CAUSAL_CHAIN = "tool_causal_chain"  # 新增：工具链因果连贯性（LLM）
    CORRECTNESS = "correctness"  # 推理正确性（LLM）


async def _invoke_eval_prompt(
    template_name: str,
    payload_data: Dict[str, object],
    user_instruction: str,
    settings: PipelineSettings,
    max_retries: int = 3
) -> str:
    """调用LLM进行评估"""
    start = time.perf_counter()
    prompt = Prompter(prompt_template=template_name)
    prompt_text = prompt.render(payload_data)
    messages = [SystemMessage(content=prompt_text), HumanMessage(content=user_instruction)]

    for attempt in range(1, max_retries + 1):
        model = await provision_chat_model(prompt_text, settings)
        try:
            response = await model.ainvoke(messages)
            content = response.content if isinstance(response.content, str) else json.dumps(response.content)
            logger.info(
                "eval_llm_call:done template=%s duration=%.2fs",
                template_name,
                time.perf_counter() - start,
            )
            return content
        except RateLimitError:
            if attempt == max_retries:
                logger.error("Rate limit exceeded for evaluation prompt: %s", template_name)
                raise
            await asyncio.sleep(2 * attempt)
        except Exception as e:
            logger.error("Error in eval LLM call: %s", e)
            if attempt == max_retries:
                raise

    raise RuntimeError(f"评估LLM调用失败: {template_name}")


def ensure_json_object(content: str) -> Dict:
    """确保内容是JSON对象"""
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {"content": parsed}
    except json.JSONDecodeError:
        # 尝试提取JSON部分
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        return {"raw_content": content}


def extract_tool_info_from_trajectory(trajectory: List[Dict]) -> List[Dict]:
    """从轨迹中提取工具调用的相关信息"""
    tool_calls_info = []

    for i, step in enumerate(trajectory):
        if step.get("role") == "tool":
            tool_info = {
                "step_id": step.get("turn_id", i + 1),
                "tool_name": step.get("tool_name"),
                "tool_args": step.get("tool_args", {}),
                "tool_observation": step.get("tool_observation", {}),
                "previous_assistant_output": None,
                "reasoning_summary": None,
                "tool_plan": None
            }

            # 查找前一个assistant步骤
            for j in range(i-1, -1, -1):
                if trajectory[j].get("role") == "assistant":
                    assistant_output = trajectory[j].get("assistant_output", {})
                    tool_info["previous_assistant_output"] = assistant_output
                    tool_info["reasoning_summary"] = assistant_output.get("reasoning_summary", "")
                    tool_plan = assistant_output.get("tool_plan", [])
                    # 找到对应的tool_plan条目
                    for plan in tool_plan:
                        if plan.get("tool_name") == tool_info["tool_name"]:
                            tool_info["tool_plan"] = plan
                            break
                    break

            tool_calls_info.append(tool_info)

    return tool_calls_info


# 优化的补充内容合理性评估函数
async def evaluate_supplementary_rationality(state: EvaluationState, config: RunnableConfig) -> dict:
    """
    评估推理过程中引入的补充内容（原文中不存在的信息）的合理性。
    包括：假设、推断、外部知识引入等是否合理、必要且逻辑连贯。
    """
    start = time.perf_counter()
    trajectory = state.get("trajectory", [])
    excerpt = state.get("excerpt", "")
    case_record = state.get("case_record", {})
    logger = logging.getLogger(__name__)

    # ===== 提取推理轨迹摘要（只提取关键信息，不判断是否补充） =====
    reasoning_summaries = []
    tool_results_summaries = []
    final_conclusion = ""

    for step in trajectory:
        role = step.get("role")
        turn_id = step.get("turn_id")

        if role == "assistant":
            assistant_output = step.get("assistant_output", {})

            # 收集reasoning_summary
            reasoning = assistant_output.get("reasoning_summary", "")
            if reasoning:
                reasoning_summaries.append({
                    "turn_id": turn_id,
                    "content": reasoning
                })

            # 收集final结论
            final = assistant_output.get("final", {})
            if isinstance(final, dict) and final.get("conclusion"):
                final_conclusion = final.get("conclusion", "")

        elif role == "tool":
            # 收集tool_results_summary
            observation = step.get("tool_observation", {})
            summary = observation.get("tool_results_summary", "")
            if summary:
                tool_results_summaries.append({
                    "turn_id": turn_id,
                    "tool_name": step.get("tool_name", ""),
                    "content": summary
                })

    # ===== 调用 LLM 评估补充内容合理性 =====
    try:
        content = await _invoke_eval_prompt(
            template_name="supplementary_rationality_evaluation",
            payload_data={
                "document_id": state.get("document_id"),
                "excerpt": excerpt,
                "problem_description": case_record.get("problem", {}).get("title", ""),
                "reasoning_summaries": json.dumps(reasoning_summaries, ensure_ascii=False, indent=2),
                "tool_results_summaries": json.dumps(tool_results_summaries, ensure_ascii=False, indent=2),
                "final_conclusion": final_conclusion,
                "domain": case_record.get("domain", "unknown")
            },
            user_instruction=("请评估推理过程中引入的补充内容的合理性。请输出JSON格式的评估结果。"),
            settings=state.get("config")
        )

        evaluation = ensure_json_object(content)

        rationality_score = max(0.0, min(1.0, float(evaluation.get("rationality_score", 0.7))))

        rationality_results = {
            "rationality_score": rationality_score,
            "rationality_level": evaluation.get("rationality_level", "中等"),
            "has_supplementary_content": evaluation.get("has_supplementary_content", False),
            "identified_supplements": evaluation.get("identified_supplements", []),
            "necessary_supplements": evaluation.get("necessary_supplements", []),
            "unnecessary_supplements": evaluation.get("unnecessary_supplements", []),
            "logical_issues": evaluation.get("logical_issues", []),
            "over_speculation_items": evaluation.get("over_speculation_items", []),
            "well_reasoned_supplements": evaluation.get("well_reasoned_supplements", []),
            "analysis": evaluation.get("analysis", ""),
            "suggestions": evaluation.get("suggestions", [])
        }

    except Exception as e:
        logger.error("Supplementary rationality evaluation failed: %s", e)
        rationality_results = {
            "rationality_score": 0.6,
            "rationality_level": "需要改进",
            "has_supplementary_content": False,
            "identified_supplements": [],
            "necessary_supplements": [],
            "unnecessary_supplements": [],
            "logical_issues": ["评估失败，无法判断逻辑问题"],
            "over_speculation_items": [],
            "well_reasoned_supplements": [],
            "analysis": "评估过程失败",
            "suggestions": ["检查补充内容合理性评估配置"]
        }

    # ===== 日志记录 =====
    logger.info(
        "evaluate_supplementary_rationality:done document_id=%s score=%.2f has_supplements=%s duration=%.2fs",
        state.get("document_id"),
        rationality_results["rationality_score"],
        rationality_results["has_supplementary_content"],
        time.perf_counter() - start,
    )

    return {
        "supplementary_rationality_results": rationality_results,
        "supplementary_rationality_score": rationality_results["rationality_score"]
    }


# 事实一致性 / 忠实性评估（LLM）
async def evaluate_factual_consistency(state: EvaluationState, config: RunnableConfig) -> dict:
    """
    评估结构化信息与推理内容是否忠实于原始文档（excerpt），
    是否存在幻觉、歪曲或无依据推断。
    改进版：全面提取 trajectory 中的 assistant、tool、tool_plan、final 等信息。
    """
    import time
    start = time.perf_counter()

    trajectory = state.get("trajectory", [])
    excerpt = state.get("excerpt", "")
    case_record = state.get("case_record", {})

    # ===== 全面提取推理中使用的事实性信息 =====
    extracted_facts = []

    for step in trajectory:
        role = step.get("role")
        turn_id = step.get("turn_id")

        if role == "assistant":
            assistant_output = step.get("assistant_output", {})

            # reasoning_summary
            reasoning = assistant_output.get("reasoning_summary")
            if reasoning:
                extracted_facts.append({
                    "type": "reasoning_summary",
                    "turn_id": turn_id,
                    "content": reasoning
                })

            # final
            final = assistant_output.get("final")
            if final:
                extracted_facts.append({
                    "type": "final_conclusion",
                    "turn_id": turn_id,
                    "content": final
                })

            # tool_plan
            tool_plan = assistant_output.get("tool_plan", [])
            for tool_entry in tool_plan:
                extracted_facts.append({
                    "type": "planned_tool",
                    "turn_id": turn_id,
                    "tool_name": tool_entry.get("tool_name"),
                    "tool_purpose": tool_entry.get("tool_purpose"),
                    "tool_args": tool_entry.get("tool_args"),
                    "action_desc": tool_entry.get("action_desc")
                })

        elif role == "tool":
            tool_name = step.get("tool_name")
            tool_args = step.get("tool_args")
            obs = step.get("tool_observation", {})

            if obs:
                extracted_facts.append({
                    "type": "tool_observation",
                    "turn_id": turn_id,
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "status": obs.get("status"),
                    "content": obs.get("results"),
                    "results_summary": obs.get("tool_results_summary")
                })

    # ===== 调用 LLM 评估模板 =====
    try:
        content = await _invoke_eval_prompt(
            template_name="factual_consistency_evaluation",
            payload_data={
                "document_id": state.get("document_id"),
                "excerpt": excerpt,
                "problem_description": case_record.get("problem", {}).get("title", ""),
                "extracted_facts": json.dumps(extracted_facts, ensure_ascii=False, indent=2)
            },
            user_instruction=(
                "请评估上述推理过程中提取和使用的结构化信息是否严格遵从原始文档事实，"
                "判断是否存在幻觉、事实歪曲、无依据推断或与原文冲突的情况。"
                "输出 JSON 格式评估结果"
            ),
            settings=state.get("config")
        )

        evaluation = ensure_json_object(content)

        consistency_score = max(
            0.0, min(1.0, float(evaluation.get("consistency_score", 0.7)))
        )

        factual_consistency_results = {
            "consistency_score": consistency_score,
            "faithfulness_level": evaluation.get("faithfulness_level", "中等"),
            "hallucinated_facts": evaluation.get("hallucinated_facts", []),
            "unsupported_claims": evaluation.get("unsupported_claims", []),
            "contradictions": evaluation.get("contradictions", []),
            "well_supported_facts": evaluation.get("well_supported_facts", []),
            "analysis": evaluation.get("analysis", ""),
            "suggestions": evaluation.get("suggestions", [])
        }

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error("Factual consistency evaluation failed: %s", e)
        factual_consistency_results = {
            "consistency_score": 0.6,
            "faithfulness_level": "未知",
            "hallucinated_facts": ["评估失败，无法判定"],
            "unsupported_claims": [],
            "contradictions": [],
            "well_supported_facts": [],
            "analysis": "评估过程失败",
            "suggestions": []
        }

    # ===== 日志记录 =====
    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        "evaluate_factual_consistency:done document_id=%s score=%.2f duration=%.2fs",
        state.get("document_id"),
        factual_consistency_results["consistency_score"],
        time.perf_counter() - start,
    )

    return {
        "factual_consistency_results": factual_consistency_results,
        "factual_consistency_score": factual_consistency_results["consistency_score"]
    }



# 1. 工具可执行性评估（规则）
async def evaluate_tool_executability(state: EvaluationState, config: RunnableConfig) -> dict:
    """评估工具调用的可执行性（基于规则）"""
    start = time.perf_counter()
    trajectory = state.get("trajectory", [])
    case_record = state.get("case_record", {})
    tools = case_record.get("tools", [])

    tool_calls_info = extract_tool_info_from_trajectory(trajectory)

    # 创建工具名称到定义的映射
    tool_definitions = {}
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            func = tool["function"]
            tool_definitions[func["name"]] = {
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
                "required": func.get("parameters", {}).get("required", [])
            }

    executability_results = []

    for tool_info in tool_calls_info:
        step_id = tool_info.get("step_id")
        tool_name = tool_info.get("tool_name")
        tool_args = tool_info.get("tool_args", {})

        if not tool_name:
            executability_results.append({
                "step_id": step_id,
                "tool_name": "unknown",
                "executable": False,
                "score": 0.0,
                "issues": ["Missing tool name"],
                "details": "工具名称为空"
            })
            continue

        # 检查工具是否定义
        if tool_name not in tool_definitions:
            executability_results.append({
                "step_id": step_id,
                "tool_name": tool_name,
                "executable": False,
                "score": 0.0,
                "issues": [f"工具 '{tool_name}' 未在可用工具列表中定义"],
                "details": f"可用工具: {list(tool_definitions.keys())}"
            })
            continue

        # 获取工具定义
        tool_def = tool_definitions[tool_name]
        required_params = tool_def.get("required", [])
        param_properties = tool_def.get("parameters", {}).get("properties", {})

        # 检查参数完整性
        param_issues = []
        missing_params = []

        for param_name in required_params:
            if param_name not in tool_args:
                missing_params.append(param_name)
            elif param_name in tool_args:
                # 简单类型检查
                param_value = tool_args[param_name]
                if param_name in param_properties:
                    param_type = param_properties[param_name].get("type", "string")

                    if param_type == "number" and not isinstance(param_value, (int, float)):
                        param_issues.append(f"参数 '{param_name}' 类型错误: 期望 {param_type}, 实际 {type(param_value).__name__}")
                    elif param_type == "boolean" and not isinstance(param_value, bool):
                        param_issues.append(f"参数 '{param_name}' 类型错误: 期望 {param_type}, 实际 {type(param_value).__name__}")
                    elif param_type == "array" and not isinstance(param_value, list):
                        param_issues.append(f"参数 '{param_name}' 类型错误: 期望 {param_type}, 实际 {type(param_value).__name__}")
                    elif param_type == "object" and not isinstance(param_value, dict):
                        param_issues.append(f"参数 '{param_name}' 类型错误: 期望 {param_type}, 实际 {type(param_value).__name__}")

        if missing_params:
            param_issues.append(f"缺少必需参数: {', '.join(missing_params)}")

        # 检查参数值合理性
        for param_name, param_value in tool_args.items():
            if param_name in param_properties:
                param_desc = param_properties[param_name].get("description", "")
                # 简单合理性检查
                if isinstance(param_value, str) and len(param_value.strip()) == 0:
                    param_issues.append(f"参数 '{param_name}' 值为空")
                elif isinstance(param_value, list) and len(param_value) == 0:
                    param_issues.append(f"参数 '{param_name}' 数组为空")

        # 计算可执行性分数
        if missing_params:
            executability_score = 0.0
            executable = False
        elif param_issues:
            executability_score = 0.5
            executable = True
        else:
            executability_score = 1.0
            executable = True

        executability_results.append({
            "step_id": step_id,
            "tool_name": tool_name,
            "executable": executable,
            "score": executability_score,
            "issues": param_issues,
            "details": {
                "required_params": required_params,
                "provided_params": tool_args,
                "missing_params": missing_params,
                "tool_definition": tool_def
            }
        })

    # 计算整体可执行性分数
    if executability_results:
        overall_score = sum(r["score"] for r in executability_results) / len(executability_results)
    else:
        overall_score = 0.0

    logger.info(
        "evaluate_tool_executability:done document_id=%s score=%.2f duration=%.2fs",
        state.get("document_id"),
        overall_score,
        time.perf_counter() - start,
    )

    return {
        "tool_executability_results": executability_results,
        "tool_executability_score": overall_score
    }


# 2. 工具工作流合理性评估（规则）
async def evaluate_tool_workflow(state: EvaluationState, config: RunnableConfig) -> dict:
    """评估工具调用流程的合理性（基于规则）"""
    start = time.perf_counter()
    trajectory = state.get("trajectory", [])

    # 提取所有工具调用步骤
    tool_steps = [step for step in trajectory if step.get("role") == "tool"]

    workflow_issues = []
    workflow_score = 1.0

    # 检查工具执行顺序的合理性
    tool_sequence = []
    for step_idx, step in enumerate(tool_steps):
        tool_sequence.append({
            "step": step.get("turn_id", step_idx + 1),
            "tool": step.get("tool_name"),
            "parameters": step.get("tool_args", {}),
            "observation": step.get("tool_observation", {})
        })

    # 1. 检查重复的工具调用
    tool_counts = {}
    for step in tool_sequence:
        tool = step["tool"]
        tool_counts[tool] = tool_counts.get(tool, 0) + 1

    redundant_tools = {tool: count for tool, count in tool_counts.items() if count > 2}
    if redundant_tools:
        workflow_issues.append({
            "type": "redundant_tool_calls",
            "description": f"工具调用过于频繁: {redundant_tools}",
            "severity": "medium",
            "steps": [s["step"] for s in tool_sequence if s["tool"] in redundant_tools]
        })
        workflow_score -= 0.1 * sum(count - 2 for count in redundant_tools.values())

    # 2. 检查工具调用的目的连贯性
    tool_categories = {
        "query": ["query_operating_conditions", "query_historical_issues", "retrieve_domain_knowledge"],
        "analyze": ["analyze_domain_principles"],
        "investigate": ["on_site_investigation"],
        "experiment": ["get_experiment_results"],
        "action": ["propose_mitigation_actions", "insert_information"]
    }

    # 检查是否有合理的工具调用模式：query -> analyze -> investigate -> experiment -> action
    category_sequence = []
    for step in tool_sequence:
        tool_name = step["tool"]
        category = next((cat for cat, tools in tool_categories.items() if tool_name in tools), "unknown")
        category_sequence.append(category)

    # 理想顺序检查
    ideal_order = ["query", "analyze", "investigate", "experiment", "action"]
    for i in range(1, len(category_sequence)):
        curr_idx = ideal_order.index(category_sequence[i]) if category_sequence[i] in ideal_order else -1
        prev_idx = ideal_order.index(category_sequence[i-1]) if category_sequence[i-1] in ideal_order else -1

        if curr_idx >= 0 and prev_idx >= 0 and curr_idx < prev_idx:
            workflow_issues.append({
                "type": "category_order_violation",
                "description": f"工具类别顺序不合理: {category_sequence[i-1]} -> {category_sequence[i]}",
                "severity": "medium",
                "steps": [tool_sequence[i-1]["step"], tool_sequence[i]["step"]]
            })
            workflow_score -= 0.1

    workflow_score = max(0.0, min(1.0, workflow_score))

    logger.info(
        "evaluate_tool_workflow:done document_id=%s score=%.2f issues=%d duration=%.2fs",
        state.get("document_id"),
        workflow_score,
        len(workflow_issues),
        time.perf_counter() - start,
    )

    return {
        "tool_workflow_results": {
            "score": workflow_score,
            "issues": workflow_issues,
            "tool_sequence": tool_sequence,
            "tool_counts": tool_counts,
            "category_sequence": category_sequence
        }
    }


# 3. 工具使用必要性评估（LLM）- 每个工具调用单独评估
async def evaluate_tool_necessity(state: EvaluationState, config: RunnableConfig) -> dict:
    """评估每个工具调用的必要性（使用LLM）"""
    start = time.perf_counter()
    trajectory = state.get("trajectory", [])
    case_record = state.get("case_record", {})
    excerpt = state.get("excerpt", "")

    tool_calls_info = extract_tool_info_from_trajectory(trajectory)

    if not tool_calls_info:
        return {"tool_necessity_results": []}

    necessity_results = []

    # 为每个工具调用创建评估任务
    for tool_info in tool_calls_info:
        step_id = tool_info.get("step_id")
        tool_name = tool_info.get("tool_name")
        tool_args = tool_info.get("tool_args", {})
        observation = tool_info.get("tool_observation", {})
        reasoning_summary = tool_info.get("reasoning_summary", "")
        tool_plan = tool_info.get("tool_plan", {})

        try:
            content = await _invoke_eval_prompt(
                template_name="tool_necessity_evaluation",
                payload_data={
                    "document_id": state["document_id"],
                    "excerpt": excerpt,
                    "problem_description": case_record.get("problem", {}).get("title", ""),
                    "step_id": step_id,
                    "tool_name": tool_name,
                    "tool_arguments": json.dumps(tool_args, ensure_ascii=False, indent=2),
                    "tool_observation": json.dumps(observation, ensure_ascii=False, indent=2),
                    "reasoning_before": reasoning_summary,
                    "tool_purpose": tool_plan.get("tool_purpose", ""),
                    "total_steps": len(tool_calls_info)
                },
                user_instruction="请评估该工具调用在当前上下文中的必要性，输出JSON格式的评估结果。",
                settings=state["config"]
            )

            # 解析评估结果
            evaluation = ensure_json_object(content)

            # 标准化评估结果
            necessity_score = max(0.0, min(1.0, float(evaluation.get("necessity_score", 0.5))))
            justification = evaluation.get("justification", "未提供理由")
            alternatives = evaluation.get("alternatives", [])
            improvement_suggestions = evaluation.get("improvement_suggestions", [])

            necessity_results.append({
                "step_id": step_id,
                "tool_name": tool_name,
                "necessity_score": necessity_score,
                "justification": justification,
                "alternatives": alternatives,
                "improvement_suggestions": improvement_suggestions,
                "necessary": necessity_score >= 0.7
            })

        except Exception as e:
            logger.error("Tool necessity evaluation failed for step %d: %s", step_id, e)
            # 创建默认评估结果
            necessity_results.append({
                "step_id": step_id,
                "tool_name": tool_name,
                "necessity_score": 0.5,
                "justification": "评估失败，使用默认值",
                "alternatives": [],
                "improvement_suggestions": [],
                "necessary": True
            })
    print("用于计算的 necessity_results:", necessity_results)
    print("长度:", len(necessity_results))
    # 计算整体必要性分数
    if necessity_results:
        overall_score = sum(r["necessity_score"] for r in necessity_results) / len(necessity_results)
    else:
        overall_score = 1.0

    logger.info(
        "evaluate_tool_necessity:done document_id=%s steps=%d score=%.2f duration=%.2fs",
        state.get("document_id"),
        len(necessity_results),
        overall_score,
        time.perf_counter() - start,
    )

    return {
        "tool_necessity_results": necessity_results,
        "tool_necessity_score": overall_score
    }


# 4. 工具最终贡献评估（LLM）- 每个工具调用单独评估
async def evaluate_tool_causality(state: EvaluationState, config: RunnableConfig) -> dict:
    """评估每个工具调用对最终结果的因果贡献（使用LLM）"""
    start = time.perf_counter()
    trajectory = state.get("trajectory", [])
    case_record = state.get("case_record", {})
    excerpt = state.get("excerpt", "")

    tool_calls_info = extract_tool_info_from_trajectory(trajectory)

    if not tool_calls_info:
        return {"tool_causality_results": []}

    causality_results = []

    # 查找最终结论
    final_conclusion = ""
    for step in reversed(trajectory):
        if step.get("role") == "assistant":
            assistant_output = step.get("assistant_output", {})
            final_output = assistant_output.get("final")
            if final_output and "conclusion" in final_output:
                final_conclusion = final_output["conclusion"]
                break

    if not final_conclusion:
        final_conclusion = "未指定最终结论"

    # 为每个工具调用创建评估任务
    for idx, tool_info in enumerate(tool_calls_info):
        step_id = tool_info.get("step_id")
        tool_name = tool_info.get("tool_name")
        tool_args = tool_info.get("tool_args", {})
        observation = tool_info.get("tool_observation", {})
        reasoning_summary = tool_info.get("reasoning_summary", "")
        tool_plan = tool_info.get("tool_plan", {})

        # 收集后续步骤信息
        subsequent_steps = []
        for subsequent in tool_calls_info[idx+1:]:
            subsequent_steps.append({
                "step_id": subsequent.get("step_id"),
                "tool_name": subsequent.get("tool_name"),
                "purpose": subsequent.get("tool_plan", {}).get("tool_purpose", "")
            })

        try:
            content = await _invoke_eval_prompt(
                template_name="tool_causality_evaluation",
                payload_data={
                    "document_id": state["document_id"],
                    "excerpt": excerpt,
                    "final_conclusion": final_conclusion,
                    "step_id": step_id,
                    "tool_name": tool_name,
                    "tool_purpose": tool_plan.get("tool_purpose", ""),
                    "tool_arguments": json.dumps(tool_args, ensure_ascii=False, indent=2),
                    "tool_results": json.dumps(observation, ensure_ascii=False, indent=2),
                    "reasoning_context": reasoning_summary,
                    "subsequent_steps": json.dumps(subsequent_steps, ensure_ascii=False, indent=2),
                    "total_steps": len(tool_calls_info)
                },
                user_instruction="请评估该工具调用对最终结论的因果贡献，输出JSON格式的评估结果。",
                settings=state["config"]
            )

            # 解析评估结果
            evaluation = ensure_json_object(content)

            # 标准化评估结果
            causality_score = max(0.0, min(1.0, float(evaluation.get("causality_score", 0.5))))
            contribution = evaluation.get("contribution_description", "未描述具体贡献")
            critical = evaluation.get("critical", causality_score >= 0.8)
            alternative_actions = evaluation.get("alternative_actions", [])
            evidence_links = evaluation.get("evidence_links", [])

            causality_results.append({
                "step_id": step_id,
                "tool_name": tool_name,
                "causality_score": causality_score,
                "contribution": contribution,
                "critical": critical,
                "alternative_actions": alternative_actions,
                "evidence_links": evidence_links
            })

        except Exception as e:
            logger.error("Tool causality evaluation failed for step %d: %s", step_id, e)
            # 创建默认评估结果
            causality_results.append({
                "step_id": step_id,
                "tool_name": tool_name,
                "causality_score": 0.5,
                "contribution": "评估失败，无法确定具体贡献",
                "critical": False,
                "alternative_actions": [],
                "evidence_links": []
            })
    print("用于计算的 causality_results:", causality_results)
    print("长度:", len(causality_results))
    # 计算整体因果贡献分数
    if causality_results:
        overall_score = sum(r["causality_score"] for r in causality_results) / len(causality_results)
    else:
        overall_score = 1.0

    logger.info(
        "evaluate_tool_causality:done document_id=%s steps=%d score=%.2f duration=%.2fs",
        state.get("document_id"),
        len(causality_results),
        overall_score,
        time.perf_counter() - start,
    )

    return {
        "tool_causality_results": causality_results,
        "tool_causality_score": overall_score
    }

# 5. 工具链因果连贯性评估（LLM）- 连续工具调用因果分析
async def evaluate_tool_causal_chain(state: EvaluationState, config: RunnableConfig) -> dict:
    """
    评估工具调用序列的因果连贯性。
    目标：判断前一个工具输出是否成为下一个工具调用的原因。
    """
    start = time.perf_counter()
    trajectory = state.get("trajectory", [])
    case_record = state.get("case_record", {})
    excerpt = state.get("excerpt", "")
    logger = logging.getLogger(__name__)

    # 提取工具调用信息
    tool_calls_info = []
    for idx, step in enumerate(trajectory):
        if step.get("role") == "tool":
            # 上一个 assistant 输出
            prev_assistant_output = trajectory[idx - 1].get("assistant_output", {}) if idx > 0 else {}
            reasoning_summary = prev_assistant_output.get("reasoning_summary", "")
            tool_plan = prev_assistant_output.get("tool_plan", [])
            # 找到当前工具对应的 plan
            tool_plan_info = next((p for p in tool_plan if p.get("tool_name") == step.get("tool_name")), {})
            tool_calls_info.append({
                "step_id": step.get("turn_id"),
                "tool_name": step.get("tool_name"),
                "tool_args": step.get("tool_args", {}),
                "tool_observation": step.get("tool_observation", {}),
                "reasoning_summary": reasoning_summary,
                "tool_plan": tool_plan_info
            })

    if len(tool_calls_info) < 2:
        return {"tool_causal_chain_results": [], "tool_causal_chain_score": 1.0}

    causal_chain_results = []

    # 对每对连续工具调用进行因果评估
    for i in range(len(tool_calls_info) - 1):
        prev_tool = tool_calls_info[i]
        next_tool = tool_calls_info[i + 1]

        step_pair_id = f"{prev_tool['step_id']}->{next_tool['step_id']}"
        prev_tool_name = prev_tool["tool_name"]
        next_tool_name = next_tool["tool_name"]
        prev_tool_results = prev_tool.get("tool_observation", {}).get("results", "")
        prev_tool_summary = prev_tool.get("tool_observation", {}).get("tool_results_summary", "")
        next_tool_args = next_tool.get("tool_args", {})
        next_tool_purpose = next_tool.get("tool_plan", {}).get("tool_purpose", "")
        reasoning_between = prev_tool.get("reasoning_summary", "") + " -> " + next_tool.get("reasoning_summary", "")

        try:
            # 调用 LLM 进行因果评估
            content = await _invoke_eval_prompt(
                template_name="tool_causal_chain_evaluation",
                payload_data={
                    "document_id": state["document_id"],
                    "excerpt": excerpt,
                    "problem_description": case_record.get("problem", {}).get("title", ""),
                    "step_pair_id": step_pair_id,
                    "prev_tool_name": prev_tool_name,
                    "prev_tool_results": prev_tool_results,
                    "prev_tool_summary": prev_tool_summary,
                    "next_tool_name": next_tool_name,
                    "next_tool_arguments": json.dumps(next_tool_args, ensure_ascii=False, indent=2),
                    "next_tool_purpose": next_tool_purpose,
                    "reasoning_between": reasoning_between
                },
                user_instruction="请评估前一个工具输出是否是下一个工具调用的原因，分析因果连贯性，输出JSON格式评估结果。",
                settings=state["config"]
            )

            evaluation = ensure_json_object(content)

            causal_score = max(0.0, min(1.0, float(evaluation.get("causal_coherence_score", 0.5))))
            causal_relation = evaluation.get("causal_relation", "未明确关系")
            evidence_of_causality = evaluation.get("evidence_of_causality", [])
            missing_links = evaluation.get("missing_causal_links", [])
            coherence_level = evaluation.get("coherence_level", "中等")
            direct_causal = evaluation.get("direct_causal", causal_score >= 0.7)
            analysis = evaluation.get("analysis", "未提供详细分析")

            causal_chain_results.append({
                "step_pair_id": step_pair_id,
                "prev_tool": prev_tool_name,
                "next_tool": next_tool_name,
                "causal_score": causal_score,
                "causal_relation": causal_relation,
                "coherence_level": coherence_level,
                "direct_causal": direct_causal,
                "evidence_of_causality": evidence_of_causality,
                "missing_links": missing_links,
                "analysis": analysis
            })

        except Exception as e:
            logger.error("Tool causal chain evaluation failed for steps %s: %s", step_pair_id, e)
            causal_chain_results.append({
                "step_pair_id": step_pair_id,
                "prev_tool": prev_tool_name,
                "next_tool": next_tool_name,
                "causal_score": 0.5,
                "causal_relation": "评估失败",
                "coherence_level": "未知",
                "direct_causal": False,
                "evidence_of_causality": [],
                "missing_links": ["评估过程失败"],
                "analysis": "评估失败，使用默认值"
            })

    # 计算整体因果连贯性分数
    if causal_chain_results:
        overall_score = sum(r["causal_score"] for r in causal_chain_results) / len(causal_chain_results)
        direct_causal_pairs = sum(1 for r in causal_chain_results if r.get("direct_causal", False))
        if direct_causal_pairs > 0:
            overall_score = min(1.0, overall_score + 0.1 * (direct_causal_pairs / len(causal_chain_results)))
    else:
        overall_score = 1.0

    logger.info(
        "evaluate_tool_causal_chain:done document_id=%s pairs=%d score=%.2f duration=%.2fs",
        state.get("document_id"),
        len(causal_chain_results),
        overall_score,
        time.perf_counter() - start,
    )

    return {
        "tool_causal_chain_results": causal_chain_results,
        "tool_causal_chain_score": overall_score
    }



# 6. 推理正确性评估（LLM）
async def evaluate_correctness(state: EvaluationState, config: RunnableConfig) -> dict:
    """评估整个推理过程的正确性（使用LLM）"""
    start = time.perf_counter()
    trajectory = state.get("trajectory", [])
    case_record = state.get("case_record", {})
    excerpt = state.get("excerpt", "")

    # 准备推理摘要
    reasoning_summary = []
    tool_calls_info = extract_tool_info_from_trajectory(trajectory)

    for tool_info in tool_calls_info:
        reasoning = tool_info.get("reasoning_summary", "")
        tool_name = tool_info.get("tool_name", "")
        step_id = tool_info.get("step_id", 0)
        if reasoning:
            reasoning_summary.append(f"步骤 {step_id} ({tool_name}): {reasoning}")

    # 查找最终结论
    final_conclusion = ""
    final_evidence = []
    final_actions = []
    for step in reversed(trajectory):
        if step.get("role") == "assistant":
            assistant_output = step.get("assistant_output", {})
            final_output = assistant_output.get("final")
            if final_output:
                final_conclusion = final_output.get("conclusion", "")
                final_evidence = final_output.get("evidence", [])
                final_actions = final_output.get("actions", [])
                break

    if not final_conclusion:
        final_conclusion = "未指定结论"

    # 准备工具调用信息
    tool_steps_info = []
    for tool_info in tool_calls_info:
        tool_steps_info.append({
            "step_id": tool_info.get("step_id"),
            "tool_name": tool_info.get("tool_name"),
            "tool_args": tool_info.get("tool_args", {}),
            "tool_results": tool_info.get("tool_observation", {})
        })

    try:
        content = await _invoke_eval_prompt(
            template_name="reasoning_correctness_evaluation",
            payload_data={
                "document_id": state["document_id"],
                "excerpt": excerpt,
                "problem_statement": case_record.get("problem", {}).get("title", ""),
                "final_conclusion": final_conclusion,
                "final_evidence": json.dumps(final_evidence, ensure_ascii=False, indent=2),
                "final_actions": json.dumps(final_actions, ensure_ascii=False, indent=2),
                "reasoning_steps": "\n".join(reasoning_summary),
                "tool_steps": json.dumps(tool_steps_info, ensure_ascii=False, indent=2),
                "total_steps": len(tool_calls_info)
            },
            user_instruction="请评估整个推理过程的正确性，输出JSON格式的评估结果。",
            settings=state["config"]
        )

        # 解析评估结果
        evaluation = ensure_json_object(content)

        # 提取正确性评估
        correctness_score = max(0.0, min(1.0, float(evaluation.get("correctness_score", 0.7))))
        logical_errors = evaluation.get("logical_errors", [])
        reasoning_gaps = evaluation.get("reasoning_gaps", [])
        strengths = evaluation.get("strengths", [])
        suggestions = evaluation.get("suggestions", [])
        evidence_quality = evaluation.get("evidence_quality", "中等")

        correctness_results = {
            "correctness_score": correctness_score,
            "correctness_level": get_correctness_level(correctness_score),
            "logical_errors": logical_errors,
            "reasoning_gaps": reasoning_gaps,
            "strengths": strengths,
            "suggestions": suggestions,
            "evidence_quality": evidence_quality,
            "overall_assessment": evaluation.get("overall_assessment", "评估完成")
        }

    except Exception as e:
        logger.error("Reasoning correctness evaluation failed: %s", e)
        # 创建默认评估结果
        correctness_results = {
            "correctness_score": 0.6,
            "correctness_level": "需要改进",
            "logical_errors": ["评估过程失败，无法确定具体错误"],
            "reasoning_gaps": [],
            "strengths": ["推理步骤完整"],
            "suggestions": ["检查评估系统配置"],
            "evidence_quality": "未知",
            "overall_assessment": "评估失败，使用默认值"
        }

    logger.info(
        "evaluate_correctness:done document_id=%s score=%.2f duration=%.2fs",
        state.get("document_id"),
        correctness_results["correctness_score"],
        time.perf_counter() - start,
    )

    return {
        "correctness_results": correctness_results
    }


def get_correctness_level(score: float) -> str:
    """根据分数确定正确性等级"""
    if score >= 0.9:
        return "优秀"
    elif score >= 0.8:
        return "良好"
    elif score >= 0.7:
        return "合格"
    elif score >= 0.6:
        return "需要改进"
    else:
        return "不合格"


# 7. 汇总评估结果
async def aggregate_evaluation_results(state: EvaluationState, config: RunnableConfig) -> dict:
    """汇总所有评估结果"""
    start = time.perf_counter()

    # 收集所有评估分数
    executability_score = state.get("tool_executability_score", 1.0)
    workflow_score = state.get("tool_workflow_results", {}).get("score", 1.0)
    necessity_score = state.get("tool_necessity_score", 1.0)
    causality_score = state.get("tool_causality_score", 1.0)
    causal_chain_score = state.get("tool_causal_chain_score", 1.0)
    correctness_score = state.get("correctness_results", {}).get("correctness_score", 1.0)
    consistency_score = state.get("factual_consistency_score", 1.0)
    rationality_score = state.get("supplementary_rationality_score", 1.0)  # 新增

    # 计算加权总分（更新权重分配）
    weights = {
        "executability": 0.08,      # 工具可执行性
        "workflow": 0.08,           # 工作流合理性
        "necessity": 0.12,          # 工具必要性
        "causality": 0.15,          # 因果贡献
        "causal_chain": 0.12,       # 工具链因果连贯性
        "correctness": 0.10,        # 推理正确性
        "factual_consistency": 0.15, # 与源文本一致性
        "supplementary_rationality": 0.20  # 新增：补充内容合理性
    }

    overall_score = (
        executability_score * weights["executability"] +
        workflow_score * weights["workflow"] +
        necessity_score * weights["necessity"] +
        causality_score * weights["causality"] +
        causal_chain_score * weights["causal_chain"] +
        correctness_score * weights["correctness"] +
        consistency_score * weights["factual_consistency"] +
        rationality_score * weights["supplementary_rationality"]  # 新增
    )

    # 确定质量等级
    if overall_score >= 0.9:
        quality_grade = "A+"
    elif overall_score >= 0.85:
        quality_grade = "A"
    elif overall_score >= 0.8:
        quality_grade = "A-"
    elif overall_score >= 0.75:
        quality_grade = "B+"
    elif overall_score >= 0.7:
        quality_grade = "B"
    elif overall_score >= 0.65:
        quality_grade = "C+"
    elif overall_score >= 0.6:
        quality_grade = "C"
    elif overall_score >= 0.5:
        quality_grade = "D"
    else:
        quality_grade = "F"

    # 生成详细的问题报告
    all_issues = []


    # 收集可执行性问题
    for result in state.get("tool_executability_results", []):
        if result["score"] < 0.8:
            all_issues.append({
                "type": "TOOL_EXECUTABILITY",
                "step": result["step_id"],
                "tool": result["tool_name"],
                "issue": f"可执行性低 ({result['score']:.2f})",
                "details": result.get("issues", [])
            })

    # 收集工作流问题
    workflow_results = state.get("tool_workflow_results", {})
    workflow_issues = workflow_results.get("issues", [])
    for issue in workflow_issues:
        all_issues.append({
            "type": "TOOL_WORKFLOW",
            "description": issue.get("description", ""),
            "severity": issue.get("severity", "medium"),
            "steps": issue.get("steps", [])
        })

    # 收集必要性低的问题
    for result in state.get("tool_necessity_results", []):
        if result["necessity_score"] < 0.6:
            all_issues.append({
                "type": "TOOL_NECESSITY",
                "step": result["step_id"],
                "tool": result["tool_name"],
                "issue": f"必要性低 ({result['necessity_score']:.2f})",
                "justification": result.get("justification", "")[:100],
                "suggested_alternatives": result.get("alternatives", [])
            })

    # 收集因果贡献低的问题
    for result in state.get("tool_causality_results", []):
        if result["causality_score"] < 0.6:
            all_issues.append({
                "type": "TOOL_CAUSALITY",
                "step": result["step_id"],
                "tool": result["tool_name"],
                "issue": f"因果贡献低 ({result['causality_score']:.2f})",
                "contribution": result.get("contribution", "")[:100]
            })

    # 收集因果连贯性问题
    for result in state.get("tool_causal_chain_results", []):
        if result["causal_score"] < 0.6:
            all_issues.append({
                "type": "TOOL_CAUSAL_CHAIN",
                "tool_pair": result["step_pair_id"],
                "prev_tool": result["prev_tool"],
                "next_tool": result["next_tool"],
                "issue": f"因果连贯性低 ({result['causal_score']:.2f})",
                "causal_relation": result.get("causal_relation", ""),
                "missing_links": result.get("missing_links", [])
            })

    # 收集推理错误
    correctness_results = state.get("correctness_results", {})
    logical_errors = correctness_results.get("logical_errors", [])
    for error in logical_errors:
        all_issues.append({
            "type": "CORRECTNESS",
            "category": "逻辑错误",
            "description": error
        })


    factual_consistency_results = state.get("factual_consistency_results", {})
    consistency_issue = {
        "type": "CONSISTENCY",
        "hallucinated_facts": [],
        "unsupported_claims": [],
        "contradictions": []
    }
    # 幻觉事实
    for fact in factual_consistency_results.get("hallucinated_facts", []):
        consistency_issue["hallucinated_facts"].append({
            "turn_id": fact.get("turn_id"),
            "description": fact.get("content", "")
        })
    # 无依据断言
    for claim in factual_consistency_results.get("unsupported_claims", []):
        consistency_issue["unsupported_claims"].append({
            "turn_id": claim.get("turn_id"),
            "description": claim.get("content", "")
        })
    # 矛盾事实
    for contradiction in factual_consistency_results.get("contradictions", []):
        consistency_issue["contradictions"].append({
            "turn_id": contradiction.get("turn_id"),
            "description": contradiction.get("content", "")
        })
    # 添加到 all_issues
    all_issues.append(consistency_issue)

    # 收集补充内容合理性问题 - 使用SUPPLEMENTARY大括号细分
    supplementary_results = state.get("supplementary_rationality_results", {})
    if supplementary_results.get("has_supplementary_content", False):

        supplementary_issues = {
            "type": "SUPPLEMENTARY_RATIONALITY",
            "category": "补充内容合理性",
            "sub_categories": {
                "unnecessary_supplements": [],
                "logical_issues": [],
                "over_speculation": []
            }
        }

        # 不必要补充
        for supplement in supplementary_results.get("unnecessary_supplements", []):
            supplementary_issues["sub_categories"]["unnecessary_supplements"].append({
                "turn_id": supplement.get("turn_id"),
                "content": supplement.get("content", "")[:150],
                "reason": supplement.get("reason", ""),
                "suggestion": supplement.get("suggestion", "")
            })

        # 逻辑问题
        for issue in supplementary_results.get("logical_issues", []):
            supplementary_issues["sub_categories"]["logical_issues"].append({
                "description": issue.get("description", ""),
                "severity": issue.get("severity", "medium"),
                "affected_turns": issue.get("affected_turn_ids", []),
                "correction": issue.get("correction_suggestion", "")
            })

        # 过度推断
        for over_spec in supplementary_results.get("over_speculation_items", []):
            supplementary_issues["sub_categories"]["over_speculation"].append({
                "turn_id": over_spec.get("turn_id"),
                "content": over_spec.get("content", "")[:150],
                "evidence_gap": over_spec.get("evidence_gap", ""),
                "risk": over_spec.get("risk_level", "中"),
                "alternative": over_spec.get("alternative_approach", "")
            })
        all_issues.append(supplementary_issues)

    # 创建总体评估报告
    overall_evaluation = {
        "document_id": state["document_id"],
        "evaluation_timestamp": time.time(),
        "overall_score": overall_score,
        "quality_grade": quality_grade,
        "dimension_scores": {
            "TOOL_EXECUTABILITY": executability_score,
            "TOOL_WORKFLOW": workflow_score,
            "TOOL_NECESSITY": necessity_score,
            "TOOL_CAUSALITY": causality_score,
            "TOOL_CAUSAL_CHAIN": causal_chain_score,  # 新增
            "CORRECTNESS": correctness_score,
            "CONSISTENCY": consistency_score,
            "SUPPLEMENTARY_RATIONALITY": rationality_score  # 新增
        },
        "dimension_details": {
            "tool_executability": state.get("tool_executability_results", []),
            "tool_workflow": state.get("tool_workflow_results", {}),
            "tool_necessity": state.get("tool_necessity_results", []),
            "tool_causality": state.get("tool_causality_results", []),
            "tool_causal_chain": state.get("tool_causal_chain_results", []),  # 新增
            "correctness": state.get("correctness_results", {}),
            "consistency": state.get("factual_consistency_results", {}),
            "supplementary_rationality": state.get("supplementary_rationality_results", {})  # 新增
        },
        "summary": f"整体评估: {quality_grade} (分数: {overall_score:.2f})",
        "weight_distribution": weights  # 添加权重信息
    }

    logger.info(
        "aggregate_evaluation_results:done document_id=%s overall_score=%.2f grade=%s duration=%.2fs",
        state.get("document_id"),
        overall_score,
        quality_grade,
        time.perf_counter() - start,
    )

    return {
        "evaluation_results": overall_evaluation
    }


# 创建评估工作流
workflow = StateGraph(EvaluationState)

# 添加评估节点
workflow.add_node("evaluate_tool_executability", evaluate_tool_executability)
workflow.add_node("evaluate_tool_workflow", evaluate_tool_workflow)
workflow.add_node("evaluate_tool_necessity", evaluate_tool_necessity)
workflow.add_node("evaluate_tool_causality", evaluate_tool_causality)
workflow.add_node("evaluate_tool_causal_chain", evaluate_tool_causal_chain)  # 新增
workflow.add_node("evaluate_correctness", evaluate_correctness)
workflow.add_node("aggregate_evaluation_results", aggregate_evaluation_results)
workflow.add_node("evaluate_factual_consistency", evaluate_factual_consistency)
workflow.add_node("evaluate_supplementary_rationality", evaluate_supplementary_rationality)


# 配置工作流连接
workflow.add_edge(START, "evaluate_tool_executability")
workflow.add_edge("evaluate_tool_executability", "evaluate_tool_workflow")
workflow.add_edge("evaluate_tool_workflow", "evaluate_tool_necessity")
workflow.add_edge("evaluate_tool_necessity", "evaluate_tool_causality")
workflow.add_edge("evaluate_tool_causality", "evaluate_tool_causal_chain")  # 新增
workflow.add_edge("evaluate_tool_causal_chain", "evaluate_factual_consistency")
workflow.add_edge("evaluate_factual_consistency", "evaluate_supplementary_rationality")  # 新增节点
workflow.add_edge("evaluate_supplementary_rationality", "evaluate_correctness")
workflow.add_edge("evaluate_correctness", "aggregate_evaluation_results")
workflow.add_edge("aggregate_evaluation_results", END)

evaluation_graph = workflow.compile()