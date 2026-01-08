import json
import logging
import time
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from graph.evaluation.cot_evaluator import evaluation_graph
from store.dataset_store import DatasetStore
from utils.settings import PipelineSettings


class EvaluationState(TypedDict, total=False):
    # 输入数据
    document_id: str
    config: PipelineSettings

    # 可选的输入数据（如果不提供，会从文件中读取）
    case_record: Optional[Dict[str, Any]]

    # 评估结果
    evaluation_results: Dict[str, Any]


logger = logging.getLogger(__name__)


def load_case_record(settings: PipelineSettings, document_id: str) -> Dict[str, Any]:
    """从dataset_store_dir中加载case_record"""
    # 构建文件路径
    file_path = Path(settings.dataset_store_dir) / f"{document_id}.jsonl"

    if not file_path.exists():
        # 尝试其他可能的扩展名
        for ext in ['.json', '.txt']:
            alt_path = Path(settings.dataset_store_dir) / f"{document_id}{ext}"
            if alt_path.exists():
                file_path = alt_path
                logger.info(f"Using alternative file: {alt_path}")
                break
        else:
            raise FileNotFoundError(f"Case record not found for document_id: {document_id}")

    try:
        # 读取整个文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if not content.strip():
            raise ValueError(f"Case record file is empty: {file_path}")

        # 清理内容：移除BOM字符
        if content.startswith('\ufeff'):
            content = content[1:]

        # 尝试解析为JSON（整个文件是一个JSON对象）
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse as single JSON, trying JSONL format: {e}")

            # 尝试作为JSONL格式解析（每行一个JSON）
            lines = content.split('\n')
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    # 尝试解析这一行
                    return json.loads(line)
                except json.JSONDecodeError as inner_e:
                    logger.debug(f"Line {line_num} is not valid JSON: {inner_e}")

            # 如果还是失败，尝试修复常见的JSON格式问题
            logger.warning("All parsing attempts failed, trying to fix JSON format")

            # 修复尾随逗号
            fixed_content = re.sub(r',\s*}', '}', content)
            fixed_content = re.sub(r',\s*]', ']', fixed_content)

            # 尝试修复属性名引号（简单处理，但可能不够完善）
            # 注意：这个修复可能会破坏字符串内容，所以只在必要时使用
            try:
                return json.loads(fixed_content)
            except json.JSONDecodeError as final_e:
                logger.error(f"Failed to parse JSON even after fixing: {final_e}")
                logger.error(f"Content preview (first 500 chars): {content[:500]}")
                raise ValueError(f"Invalid JSON in case record file: {file_path} - {final_e}")

    except Exception as e:
        logger.error(f"Unexpected error loading case record from {file_path}: {e}")
        raise RuntimeError(f"Failed to load case record from {file_path}: {e}")


def load_excerpt_from_cache(settings: PipelineSettings, document_id: str) -> str:
    """从cache目录加载excerpt"""
    # 构建缓存文件路径: cache/{document_id}/full_text.txt
    cache_dir = Path(settings.cache_dir) / document_id
    excerpt_path = cache_dir / "full_text.txt"

    # 如果full_text.txt不存在，尝试其他可能的文件名
    if not excerpt_path.exists():
        # 尝试其他可能的文件名
        possible_files = [
            cache_dir / "excerpt.txt",
            cache_dir / "summary.txt",
            cache_dir / "text.txt",
        ]

        for file_path in possible_files:
            if file_path.exists():
                excerpt_path = file_path
                logger.info(f"Using alternative excerpt file: {excerpt_path}")
                break
        else:
            # 如果没有找到任何文件，记录警告并返回空字符串
            logger.warning(f"Excerpt file not found for document_id: {document_id}")
            return ""

    try:
        with open(excerpt_path, 'r', encoding='utf-8') as f:
            excerpt = f.read().strip()

        logger.debug(f"Loaded excerpt from {excerpt_path}, length: {len(excerpt)}")
        return excerpt

    except Exception as e:
        logger.error(f"Failed to load excerpt from {excerpt_path}: {e}")
        return ""


def extract_metadata_from_case_record(case_record: Dict[str, Any]) -> Dict[str, str]:
    """从case_record中提取metadata"""
    metadata = {}

    # 基础信息
    if 'id' in case_record:
        metadata['id'] = str(case_record['id'])

    if 'domain' in case_record:
        metadata['domain'] = str(case_record['domain'])

    if 'language' in case_record:
        metadata['language'] = str(case_record['language'])

    # meta字段
    if 'meta' in case_record and isinstance(case_record['meta'], dict):
        for key, value in case_record['meta'].items():
            metadata[f'meta_{key}'] = str(value)

    # problem字段
    if 'problem' in case_record and isinstance(case_record['problem'], dict):
        problem = case_record['problem']

        if 'title' in problem:
            metadata['problem_title'] = str(problem['title'])

        if 'context' in problem and isinstance(problem['context'], dict):
            context = problem['context']
            for key, value in context.items():
                metadata[f'context_{key}'] = str(value)

    # labels字段
    if 'labels' in case_record and isinstance(case_record['labels'], dict):
        labels = case_record['labels']
        for key, value in labels.items():
            metadata[f'label_{key}'] = str(value)

    return metadata


async def run_evaluation(state: EvaluationState, config: RunnableConfig) -> dict:
    """评估案例 → 对生成的案例进行质量评估"""
    start = time.perf_counter()
    document_id = state.get("document_id", "unknown")
    logger.info("evaluation:start document_id=%s", document_id)

    try:
        # 获取设置
        settings = state["config"]

        # 加载case_record（如果state中没有提供）
        case_record = state.get("case_record")
        if not case_record:
            try:
                case_record = load_case_record(settings, document_id)
            except Exception as e:
                logger.error(f"Failed to load case record for {document_id}: {e}")
                return {
                    "evaluation_results": {
                        "overall_score": 0.0,
                        "error": f"Failed to load case record: {str(e)}",
                        "document_id": document_id,
                        "timestamp": time.time()
                    }
                }

        # 从缓存中加载excerpt
        excerpt = load_excerpt_from_cache(settings, document_id)

        # 如果缓存中没有excerpt，尝试从case_record中提取
        if not excerpt:
            logger.warning(f"No excerpt found in cache for {document_id}, trying case_record")
            # 尝试从case_record中提取
            if 'problem' in case_record and isinstance(case_record['problem'], dict):
                problem = case_record['problem']
                if 'user_report' in problem:
                    excerpt = str(problem['user_report'])


        # 从case_record中获取trajectory
        trajectory = case_record.get('trajectory', [])

        logger.debug(
            "evaluation:loaded_data document_id=%s excerpt_len=%d metadata_keys=%s trajectory_len=%d",
            document_id,
            len(excerpt),
            len(trajectory)
        )

        # 调用评估子图
        eval_state = await evaluation_graph.ainvoke(
            {
                "document_id": document_id,
                "config": settings,
                "case_record": case_record,
                "excerpt": excerpt,
                "trajectory": trajectory,
            }
        )

        evaluation_results = eval_state.get("evaluation_results", {})

    except Exception as e:
        logger.error(f"Evaluation failed for {document_id}: {e}", exc_info=True)
        evaluation_results = {
            "overall_score": 0.0,
            "error": f"Evaluation failed: {str(e)}",
            "document_id": document_id,
            "timestamp": time.time()
        }

    duration = time.perf_counter() - start
    logger.info(
        "evaluation:done document_id=%s duration=%.2fs",
        document_id,
        duration
    )

    return {
        "evaluation_results": evaluation_results,
    }


async def persist_evaluation_results(state: EvaluationState, config: RunnableConfig) -> dict:
    """保存评估结果"""
    start = time.perf_counter()
    document_id = state.get("document_id", "unknown")

    try:
        settings = state["config"]

        # 确保评估目录存在
        eval_dir = Path(settings.dataset_store_eval)
        eval_dir.mkdir(parents=True, exist_ok=True)

        # 创建DatasetStore实例
        dataset_eval_store = DatasetStore(eval_dir)

        # 获取评估结果
        eval_record = state.get("evaluation_results")
        if not eval_record:
            logger.warning(f"No evaluation results found for {document_id}, saving empty result")
            eval_record = {
                "document_id": document_id,
                "error": "No evaluation results generated",
                "timestamp": time.time()
            }

        # 保存评估结果
        dataset_eval_store.append(document_id, [eval_record])

        logger.info(
            "persist_evaluation_results:done document_id=%s duration=%.2fs file=%s",
            document_id,
            time.perf_counter() - start,
            eval_dir / f"{document_id}.jsonl"
        )

    except Exception as e:
        logger.error(f"Failed to persist evaluation results for {document_id}: {e}", exc_info=True)
        # 重新抛出异常，让工作流知道保存失败
        raise

    return {}


# 创建评估工作流
evaluation_workflow = StateGraph(EvaluationState)
evaluation_workflow.add_node("run_evaluation", run_evaluation)
evaluation_workflow.add_node("persist_evaluation_results", persist_evaluation_results)

# 配置工作流连接 - 简化版：评估后直接保存
evaluation_workflow.add_edge(START, "run_evaluation")
evaluation_workflow.add_edge("run_evaluation", "persist_evaluation_results")
evaluation_workflow.add_edge("persist_evaluation_results", END)

evaluation_pipeline = evaluation_workflow.compile()
