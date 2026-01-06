## 目录与复用约定

- `graph/` 只放 LangGraph 相关代码：主图 `pipeline.py` 负责编排，复用逻辑必须拆到 `graph/subgraphs/`（目前有 `pdf_parsing.py`、`case_builder.py`），方便新流水线直接复用/扩展子图，而不是修改主图。
- `prompts/` 统一管理所有提示词，使用 `ai-prompter` + Jinja（参考 `prompts/case_outline.jinja`、`case_trajectory.jinja`）。任何 prompt 变更都在这里调整，禁止在代码中硬编码。
- `store/` 封装所有持久化逻辑（文档、数据集、缓存）。若后续要接数据库/向量库，优先在这一层扩展，避免在 LangGraph 节点里直接操作外部资源。
- `utils/` 放配置、LLM、PDF 解析等通用工具，所有可调参数都应走 `.env`/`PipelineSettings`。当前支持 `OPENAI_CONTEXT_WINDOW`（控制解析/提示词的上下文长度）和 `OPENAI_TIMEOUT`（LLM 请求超时），调试新模型时务必同步修改。

## 开发与运行要求

- 运行python脚本时，应使用 `uv run xxx.py` 来运行。
- 日常测试、批处理或服务均应通过脚本/命令层调用 LangGraph，保持 open-notebook 式的“命令 → 图”解耦。CLI/服务端不要直接内联图逻辑。
- 本地调试必须使用 `scripts/run_sample.sh`（或基于该脚本包装的命令）来运行，脚本会清空 Clash 注入的 `*_proxy` 环境变量，防止 OpenAI 兼容接口被错误代理。不要直接 `python main.py`。
- 子图/节点必须定义清晰的 `TypedDict` 状态、重试策略以及输出校验（例如 `ensure_json_object`）。如果需要新增节点/子图，保持这种模式以便复用和排障。
- 提示词调优、LangGraph 结构优化请参考 `_sample/open-notebook-main` 的做法：拆子图、集中 prompt、命令层入口。如果要新增 pipeline，先评估是否可以复用现有子图和 prompts，再考虑新建。
