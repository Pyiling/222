# 数据流水线

该项目实现了一条 **PDF → 文本 → SFT 数据** 的 LangGraph 流水线，输出 JSON 结构与 `output_example.json` 完全一致，可直接用于大模型监督微调。核心目录如下：

- `graph/`：主图 `pipeline.py` 只负责编排；可复用的子图收敛到 `graph/subgraphs/`，例如解析子图 `pdf_parsing.py`、案例构建子图 `case_builder.py`。
- `prompts/`：所有提示词都使用 `ai-prompter` + Jinja 管理，并统一使用中文，便于集中调优和复用。
- `store/`：轻量的文件存储（文档、数据集、缓存），与图逻辑解耦。
- `utils/`：公共工具（配置、PDF 解析、LLM 接入、文本处理等）。

## 工作流阶段

1. **Parse 子图** (`graph/subgraphs/pdf_parsing.py`)
   - `parse_pdf`：按页解析 PDF（默认 1 页为一个 batch），落地到 `data/cache/<document_id>/batch_XXXX.json`；摘取的摘要长度受 `.env` 中 `OPENAI_CONTEXT_WINDOW` 限制，避免超过模型上下文。
   - `persist_document`：将全文与元数据写入 `data/documents/<document_id>.json`。
2. **Case Builder 子图** (`graph/subgraphs/case_builder.py`)
   - `prepare_context`：根据上下文窗口裁剪文本，为 LLM 构建高置信片段。
   - `draft_case_outline`：通过 `prompts/case_outline.jinja` 生成包含 domain/meta/problem/labels 的案例蓝图。
   - `extract_toolset`：通过 `prompts/toolset_plan.jinja` 提取工具清单，或使用 `TOOLSET_NAME` 指定固定工具集文件。
   - `step_extraction`：通过 `prompts/steps_extract.jinja` 梳理排查步骤（允许缺失）。
   - `step_completion`：通过 `prompts/steps_complete.jinja` 补齐缺失的步骤信息。
   - `trajectory_builder`：通过 `prompts/trajectory_step.jinja` / `prompts/trajectory_final.jinja` 生成轨迹。
   - `finalize_case_record`：整合蓝图与轨迹，得到符合 `output_example.json` 的单条案例。
3. **persist_dataset**：把案例记录（多行 JSON）追加到 `data/datasets/<document_id>.jsonl`，方便人工查阅和下游训练。

各节点内置 RateLimitError 重试逻辑，全部提示词均为中文描述，确保生成的数据与原始 PDF 语境匹配。

## 环境准备与配置

```bash
uv sync
cp .env.example .env  # 填写 OpenAI 兼容 API 信息
```

`.env` 关键字段：

- `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`：OpenAI 兼容接口配置。
- `OPENAI_CONTEXT_WINDOW`：LLM 上下文窗口（默认 120000 tokens），解析与提示词阶段都会按此窗口裁剪输入。
- `OPENAI_TIMEOUT`：HTTP/流式调用的整体超时（秒），默认 30，可按模型稳定性调整。
- `DOCUMENT_STORE_DIR`、`DATASET_STORE_DIR`、`CACHE_DIR`：三类持久化目录，可按需自定义。
- `PDF_BATCH_SIZE`：PDF 按页切分的 batch 大小（默认 1 页，防止 OOM）。
- `TOOLS_DIR`：固定工具集所在目录（默认 `data/tools`）。
- `TOOLSET_NAME`：可选。指定工具集 JSON 文件名（如 `toolset.json`），位于 `TOOLS_DIR` 下。
- `OCR_API_URL` / `OCR_API_KEY` / `OCR_MODEL` / `OCR_TIMEOUT`：OCR 解析使用的本地 vLLM 配置。

仓库内置示例工具集：`data/tools/sample_toolset.json`。

## 运行示例

仓库自带 `data/documents/sample_0.pdf`。默认 `document_id=sample_0`（取文件名），也可通过 `--document-id` 自定义。

```bash
# 默认示例（document_id=sample_0，PDF 解析）
scripts/run_sample.sh data/documents/sample_0.pdf

# OCR 解析模式
scripts/run_sample.sh data/documents/sample_0.pdf --parse-mode ocr

# 自定义 PDF 与 document_id
scripts/run_sample.sh data/documents/sample_0.pdf --document-id demo_case
```

执行完成后，可以在以下目录查看结果：

- `data/cache/<document_id>/batch_*.json`：逐页缓存的解析文本；
- `data/documents/<document_id>.json`：聚合后的全文与元数据；
- `data/datasets/<document_id>.jsonl`：单条案例记录，字段布局与 `output_example.json` 保持一致。

若需要扩展其它流水线，可直接复用 `graph/subgraphs/` 中的子图，并在 `prompts/` 目录新增提示词模板即可。
