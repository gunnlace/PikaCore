# PikaCore 当前架构

[English](PIKACORE_DESIGN.md) | [简体中文](PIKACORE_DESIGN_CN.md)

英文文档是 canonical version。功能文档发生变化时，必须在同一个 pull request 中同步更新英文版和简体中文版。

本文档描述 PikaCore 0.1.0 已实现的行为。它是实现参考，不是 roadmap。

PikaCore fork 自 [CoreCoder](https://github.com/he-yufeng/CoreCoder)。PikaCore 保留了 CoreCoder 紧凑的 Python package 和原生 function-calling loop，并增加了 workspace boundary、permission-aware tool execution、durable state、恢复、Working Memory、上下文压缩、CLI state view 和离线 fixture benchmark。原 CoreCoder MIT attribution 保留在仓库的 `LICENSE` 文件中。

## 架构

```text
CLI / one-shot prompt
        │
        ▼
Agent ───────────────► LLM / LiteLLM
  │                       │
  │ native tool calls     │ streamed text + structured calls
  ▼                       │
ToolExecutor ◄────────────┘
  │
  ├── PermissionPolicy + main-thread approval
  ├── read batches in parallel
  └── write / bash / agent barriers in order
        │
        ▼
WorkspaceContext + built-in tools
        │
        ├── structured ToolExecutionResult
        └── file freshness / workspace delta

Agent durability points
  ├── ProjectStore: SessionState, RunState, Checkpoint, Report, TraceEvent
  ├── WorkingMemoryManager: structured runtime events only
  └── ContextManager: protocol-safe layered compression
```

### 组件职责

| Component | 已实现职责 |
|---|---|
| `cli.py` | CLI arguments、REPL rendering、终端审批和 streaming display。 |
| `commands.py` | 纯本地命令解析，以及对 Agent/Store-backed API 的调用。 |
| `Agent` | Model/tool loop、tool-call pairing、run lifecycle、durability 和 recovery wiring。 |
| `ToolExecutor` | Tool lookup、argument binding、permission decision、approval、scheduling 和 structured results。 |
| `WorkspaceContext` | Git-root discovery、canonical path resolution、workspace snapshot 和 fingerprint。 |
| `ProjectStore` | Atomic JSON、redacted JSONL、schema loading 和完整的 named session snapshot。 |
| `WorkingMemoryManager` | 根据 user、tool、checkpoint、recovery 和 run event 执行确定性、有容量上限的更新。 |
| `ContextManager` | Token estimation、safe protocol split、layered compression 和 `CompressionResult`。 |
| `benchmarks/` | 隔离的 fixture repository、`ScriptedFakeLLM`、ablation、确定性 outcome view 和 JSON report。 |

## Agent 与工具协议

每个 `Agent` 都拥有自己的工具实例和 tool-name lookup table。每个 assistant tool call 后必须恰好有一个相同 ID 的 tool result。工具并行执行不会改变 model 看到的结果顺序。

内置工具集为：

- `read_file`
- `write_file`
- `edit_file`
- `glob`
- `grep`
- `bash`
- `agent`

连续的 read-only call 作为一个并行 batch 执行。每个 write、edit、shell 或 sub-agent call 都形成串行 barrier。同一组中，如果某个 barrier 被拒绝，后续 barrier call 也会被拒绝。Tool completion 统一为 `ToolExecutionResult`，其中包含 status、error code、approval outcome、exit code、read paths、affected paths、workspace-change flag、truncation 和 duration。

Sub-agent 拥有独立的 conversation 和 run identity，共享 parent model、workspace、store 和 permission policy，并且不会收到 `agent` 工具。这是当前实现的 recursion guard。

## Workspace 与权限边界

Workspace 是当前所在的 Git repository；如果 Git discovery 失败，则使用启动目录。文件参数在执行前会 canonicalize。Resolver 会拒绝 traversal、root 外路径、symlink escape，以及将 workspace root 本身作为 write target。对于不存在的 write target，只有在其已存在的 parent chain 解析后仍位于 workspace 内部时才会接受。

Permission mode 按工具进行判断：

| Mode | Read-only tool | Mutating/high-risk tool |
|---|---|---|
| `read-only` | allow | deny |
| `ask` | allow | 在主线程调用 CLI approval callback |
| `auto` | allow | allow |

`bash` 被标记为 high-risk 和 mutating。它会阻止一小组已知 destructive command pattern，从 workspace 开始运行，按 BashTool instance 维护 cwd，捕获真实 exit code，并截断超大的 display output。其 subprocess environment 使用明确的 cross-platform allowlist，且排除包括 API key 在内的 credential-like variables。

这些 application control 不是 operating-system sandbox。准确边界和限制见 [SECURITY_CN.md](SECURITY_CN.md)。

## Durable state

State schema version 1 存放在当前项目下：

```text
.pikacore/
├── sessions/<session_id>.json
├── runs/<run_id>/task_state.json
├── runs/<run_id>/trace.jsonl
├── runs/<run_id>/report.json
└── checkpoints/<checkpoint_id>.json
```

`SessionState` 包含 structured messages、Working Memory、model、repository root、checkpoint link 和 run IDs。`RunState` 记录一次 user request 或会改变状态的 CLI operation。`Report` 在 run 结束时固化 model/tool/token/compression/approval/path metrics。`TraceEvent` 以 append-only JSONL 记录固定 event vocabulary。

Session、run、checkpoint 和 report 文件使用 atomic replacement。Trace line 以 append 方式写入并 flush。持久化前会执行递归 secret redaction；Session 字符串不会因长度而截断，而 trace/report summary 使用有边界的字符串。损坏的最后一条 trace line 会被忽略并返回 warning；更早行的损坏会作为错误报告。

命名 `/save` snapshot 会复制完整 `SessionState`，并使用 snapshot 的新 session identity 克隆关联 checkpoint。CLI 不直接打开这些文件。

### Durability points

Agent 在 message boundary 和恢复所需的 operation 周围执行持久化。任何工具 side effect 开始前，assistant tool calls 都必须已写入 checkpoint。Tool result 在 loop 继续之前完成保存和 pairing。Read-batch completion 和 mutating barrier 会创建 checkpoint。手动 context compaction 以及 model/permission change 也通过同一套 run/report/checkpoint lifecycle 执行。

如果 required checkpoint 无法保存，Agent 不会启动后续 side-effecting tool。Persistence failure 会作为 warning 暴露；如果 report 可写，还会记录在 report metrics 中。

## 恢复语义

可以通过 `-r/--resume` 和 `/session resume <id>` 恢复。当前 runtime identity 包含：

- model；
- canonical repository root；
- 当前 Git branch；
- 汇总为一个 signature 的 tool name、risk/read-only flag 和 schema；
- permission mode；
- state schema version。

Checkpoint file freshness 保存相关 read 或 modified path 的 hash。目录形式的 `grep` 和 `glob` search 会被保守标记为 unverifiable，因此 result set 可能变化时，恢复会要求重新检查。

恢复返回以下五类结果之一：

| Status | 含义与行为 |
|---|---|
| `full-valid` | Runtime identity 和保存的 file fingerprint 仍匹配。 |
| `files-stale` | 文件已变化、消失，或目录 search 无法验证；追加 notice 并要求重新读取。 |
| `runtime-mismatch` | Model、root、branch、tools、permission mode 或 checkpoint presence 不同；追加 review notice。 |
| `incomplete-tool-call` | Assistant call 没有 result；补充恰好一个 interrupted result，并要求检查 workspace。 |
| `schema-mismatch` | Persisted schema 不受支持；不恢复，也不修改保存的 state。 |

任何 pending tool call 都不会自动重放，read call 也不例外。状态未知的既有 write、edit、shell 或 sub-agent execution 会表示为 interrupted；recovery notice 会要求 model 和用户先检查 workspace，再决定是否 retry。

## Working Memory

Working Memory 是 `SessionState` 的一部分，不使用独立 memory directory。它包含 current request、紧凑的 task summary、已记忆文件及 freshness、10 条 recent shell commands、最多 10 个 blockers 和最多 10 个 next steps。File memory 上限为 30 条。Task/file/item 字符串也有明确的长度限制。

更新只能来自结构化 user、tool、checkpoint、recovery 和 run event。Final-answer prose 不会因为固定句式而被解析。成功 reread 会刷新 file hash 并解除对应的 stale-file blocker。`/memory clear` 经确认后只清空 Working Memory；`/reset` 会清空 messages、Working Memory 和 recovery continuity。

## 上下文压缩

`ContextManager` 估算 token pressure，并按以下顺序应用已实现的 layer：

1. snip old tool output；
2. merge duplicate reads；
3. extract old search and command material；
4. 必要时通过配置的 LLM summarize old turns；
5. 在 hard pressure 下 collapse older turns。

Tool-call/result group 保持 structured，`_safe_split` 不会切断 active protocol group。Working Memory 和 old turns 使用独立的 summary-input budget，因此较大的 memory 不会排除全部 conversation history。每次发生变化的 compression 都会产生 `CompressionResult`，并在 trace/report state 中记录 strategy、before/after token estimate 和 message count。`/compact` 使用这条 Agent-level durable path。

## CLI state window

本地 command router 只调用稳定的 Agent 和 Store API，不直接解析 JSON。已实现命令为：

```text
/help
/reset
/memory
/memory files
/memory clear
/session
/session list
/session new
/session resume <id>
/sessions
/runs [n]
/trace [run_id] [n]
/permissions [read-only|ask|auto]
/tokens
/model [name]
/compact
/diff
/save [name]
quit
```

`/tokens` 聚合当前 session 的 report totals。`/diff` 使用归因于 structured tool result 和 report、且限定在当前 session 中的路径。Command parsing 不调用 model，并接受可注入的 confirmation callback 以便测试。

## 离线评估

Evaluation harness 只会将 manifest 中列出的 regular fixture file 放入 temporary directory。`ScriptedFakeLLM` 提供确定性 response；timestamp、duration、UUID 和其他非确定字段不参与 outcome comparison。仅支持 baseline、Working Memory off 和 context policy off 三种 variant。

已实现 fixture 覆盖 edit success、bad-argument retry、path escape、permission rejection、parallel reads、write barriers、large output、stale-file resume、unknown-write resume 和 Working Memory freshness。当前实测结果和 runner command 见 [BENCHMARKS.md](BENCHMARKS.md)。
