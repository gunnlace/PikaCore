# PikaCore

[English](README.md) | [简体中文](README_CN.md)

英文文档是 canonical version。功能文档发生变化时，必须在同一个 pull request 中同步更新英文版和简体中文版。

PikaCore 是一个小型编码 Agent harness，支持原生 function calling、项目本地状态、可恢复运行、权限控制、Working Memory、上下文压缩和离线评估套件。

PikaCore fork 自 [CoreCoder](https://github.com/he-yufeng/CoreCoder)。原 CoreCoder 版权和 MIT License 声明完整保留在 [LICENSE](LICENSE) 中；PikaCore 继续使用 MIT License 分发。

## 已实现功能

- 支持流式输出的 OpenAI-compatible 模型访问，以及可选的 LiteLLM 模型访问。
- 原生工具调用：文件读取、写入、编辑、搜索、shell 命令和不可递归的 sub-agent。
- 文件工具的仓库路径边界、三种权限模式、主线程审批、经过净化的 shell 环境，以及结构化工具结果。
- 读取批次并行执行；写入、shell 和 sub-agent barrier 串行执行；结果保持模型请求时的顺序。
- 在当前仓库的 `.pikacore/` 目录下保存原子 session、run、checkpoint、report JSON，以及脱敏的 JSONL trace。
- 五类恢复结果、文件 freshness 检查，以及不自动重放 pending tool call 的 interrupted result 修复。
- 有容量上限、由事件驱动的 Working Memory，以及可观测的分层上下文压缩。
- 由 `ScriptedFakeLLM` 驱动的 10 个确定性 fixture benchmark；benchmark runner 默认不调用真实 provider。

详细契约请参阅[当前架构](docs/PIKACORE_DESIGN_CN.md)、[安全边界](docs/SECURITY_CN.md)和 [Benchmark 结果](docs/BENCHMARKS.md)。

## 环境要求

- Python 3.10–3.13
- 使用 `uv` 完成本文档中的安装和开发流程
- 交互式或 one-shot 模型调用需要 API key；离线 benchmark 和测试套件不需要 API key

## 从源码安装

PikaCore 尚未配置 PyPI 发布。请从本仓库安装：

```bash
git clone https://github.com/gunnlace/PikaCore.git
cd PikaCore
uv sync
uv run pikacore --version
```

安装开发工具：

```bash
uv sync --extra dev
```

安装可选的 LiteLLM backend：

```bash
uv sync --extra litellm
```

## 配置模型

PikaCore 优先读取 `PIKACORE_*` 环境变量。`OPENAI_API_KEY` 和 `OPENAI_BASE_URL` 保留其常规含义，`CORECODER_*` 环境变量仅作为兼容 fallback。

| 配置项 | 解析顺序 |
|---|---|
| API key | `PIKACORE_API_KEY`, `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `CORECODER_API_KEY` |
| Model | `PIKACORE_MODEL`, `CORECODER_MODEL`, 然后是 `gpt-5.5` |
| Base URL | `PIKACORE_BASE_URL`, `OPENAI_BASE_URL`, `CORECODER_BASE_URL` |
| Output limit | `PIKACORE_MAX_TOKENS`, `CORECODER_MAX_TOKENS`, 然后是 `4096` |
| Context limit | `PIKACORE_MAX_CONTEXT`, `CORECODER_MAX_CONTEXT`, 然后是 `128000` |
| Temperature | `PIKACORE_TEMPERATURE`, `CORECODER_TEMPERATURE`, 然后是 `0` |
| Backend | `PIKACORE_PROVIDER`, `CORECODER_PROVIDER`, 然后是 `openai` |

```bash
export OPENAI_API_KEY=sk-...
export PIKACORE_MODEL=gpt-5.5
uv run pikacore
```

使用其他 OpenAI-compatible endpoint：

```bash
export OPENAI_API_KEY=your-key
export PIKACORE_BASE_URL=https://api.example.com/v1
export PIKACORE_MODEL=provider-model
uv run pikacore
```

安装 `litellm` extra 后，通过 `PIKACORE_PROVIDER=litellm` 选择该 backend。

对应的 CLI 参数为 `--api-key`、`--base-url` 和 `--model`。配置优先级依次为 CLI 参数、主环境变量、兼容 fallback、内置默认值。

## 运行 PikaCore

在希望 PikaCore 操作的仓库中启动交互式 REPL：

```bash
uv run pikacore --permissions ask
```

执行一次请求后退出：

```bash
uv run pikacore --permissions read-only -p "Explain the failing tests"
```

恢复已保存的 session：

```bash
uv run pikacore -r session_id
```

PikaCore 会将所在的 Git 仓库识别为 workspace。在 Git 仓库之外，当前目录将作为 workspace root。

## 权限模式

默认模式为 `ask`。

| Mode | Read-only tools | Write, edit, shell, sub-agent |
|---|---|---|
| `read-only` | 允许 | 拒绝 |
| `ask` | 允许 | 需要终端审批 |
| `auto` | 允许 | 无需审批即可执行 |

通过 `--permissions read-only|ask|auto` 选择初始模式，也可以使用 `/permissions` 查看或修改当前进程的模式。运行时变更会记录到 run trace 和 checkpoint identity 中。

`auto` 不是 filesystem 或 process sandbox。在不可信任务上启用前，请先阅读[安全边界](docs/SECURITY_CN.md)。

## 本地命令

| Command | 行为 |
|---|---|
| `/help` | 显示命令帮助。 |
| `/reset` | 清空 conversation messages、Working Memory 和 recovery continuity。 |
| `/memory` | 显示当前有容量上限的 Working Memory。 |
| `/memory files` | 显示已记忆的文件、action 和 freshness。 |
| `/memory clear` | 确认后清空 Working Memory，但不清空 messages。 |
| `/session` | 显示当前 session metadata。 |
| `/session list` | 列出项目中最近的 session。 |
| `/session new` | 保存当前 session 并创建一个空 session。 |
| `/session resume <id>` | 验证 recovery state 后切换 session。 |
| `/sessions` | `/session list` 的兼容 alias。 |
| `/runs [n]` | 显示当前 session 最近的 run；默认 10。 |
| `/trace [run_id] [n]` | 显示最近的脱敏事件；默认当前 run 和 20 条事件。 |
| `/permissions [mode]` | 显示工具风险，或设置 `read-only`、`ask`、`auto`。 |
| `/tokens` | 从当前 session 的 report 聚合 token 用量及已知模型的 cost。 |
| `/model [name]` | 显示或修改 model，并更新 checkpoint runtime identity。 |
| `/compact` | 通过 durable Agent lifecycle 执行上下文压缩。 |
| `/diff` | 显示当前 session 中归因于工具结果的路径。 |
| `/save [name]` | 保存完整的命名 SessionState snapshot；名称含空格时需要加引号。 |
| `quit` | 退出 REPL。 |

本地命令只调用 Agent 和 Store API；CLI command router 不直接读写状态 JSON。

## 状态目录与恢复

Runtime state 按当前仓库隔离：

```text
.pikacore/
├── sessions/<session_id>.json
├── runs/<run_id>/task_state.json
├── runs/<run_id>/trace.jsonl
├── runs/<run_id>/report.json
├── checkpoints/<checkpoint_id>.json
└── benchmarks/phase6-report.json   # created when benchmarks run
```

Git 会忽略 `.pikacore/`。REPL input history 单独存放在 `~/.pikacore_history`，本仓库同样会忽略该文件。

恢复时，PikaCore 会比较 checkpoint 与当前 model、repository、branch、tool schema、permission mode 和文件 fingerprint。恢复结果分为 `full-valid`、`files-stale`、`runtime-mismatch`、`incomplete-tool-call` 和 `schema-mismatch`。任何 pending tool call 都不会自动重放。没有配对结果的 call 会收到 interrupted tool result；对于 mutating 或状态未知的既有执行，还会附加要求检查 workspace 的提示。详见[当前架构](docs/PIKACORE_DESIGN_CN.md)。

## 离线 Benchmark

运行 baseline 和两个已支持的 ablation：

```bash
uv run python benchmarks/run_benchmarks.py --ablation all
```

当前确定性套件包含 10 个 fixture 和 3 个 variant。2026-08-12 的结果为：baseline 通过 10/10；Working Memory off 通过 9/10；context policy off 通过 9/10。30 个 outcome 的 `completed` 均为 true，共 28/30 通过检查。两个 ablation failure 是预期的 feature-sensitivity check。完整方法和各 variant 结果见 [Benchmark 结果](docs/BENCHMARKS.md)。

## 开发检查

```bash
uv lock --check
uv run --extra dev ruff check .
uv run --extra dev pytest tests/ -q
uv run python -m compileall pikacore
```

## 安全摘要

文件工具会拒绝 `..` traversal、workspace 外部的 absolute path 和 symlink escape。Shell subprocess 只接收 allowlist 中的环境变量，其中不包含 API key 和其他 credential-like variables。State 和 trace 会在持久化前递归脱敏，trace 字符串上限为 4,000 个字符。

这些控制可以降低意外泄漏和跨仓库写入的风险，但它们不是 OS sandbox。Shell 命令仍然可以访问当前用户有权访问的资源；model request 会把选定的 context 发送给配置的 provider；本地 state 未加密。处理不可信工作时应使用 `read-only` 或 `ask`，并将 `.pikacore/` 视为私有数据。完整边界见[安全边界](docs/SECURITY_CN.md)。

## License 与上游归属

PikaCore fork 自 [CoreCoder](https://github.com/he-yufeng/CoreCoder)，原作者为 copyright © 2026 Yufeng He。上游 MIT copyright 和 permission notice 已逐字保留在 [LICENSE](LICENSE) 中。PikaCore 的修改同样使用 MIT License。
