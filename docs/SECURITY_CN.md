# PikaCore 安全边界

[English](SECURITY.md) | [简体中文](SECURITY_CN.md)

英文文档是 canonical version。功能文档发生变化时，必须在同一个 pull request 中同步更新英文版和简体中文版。

PikaCore 会在用户机器上执行 model 选择的工具。它的控制可以降低意外 workspace escape、未审批 mutation 和 credential leakage 的风险，但不能把不可信 model response 或 shell command 变成 sandboxed code。

## Trust model

用户选择 repository、model provider、prompt、permission mode 和 approval decision。配置的 model provider 会收到 conversation content 和选定的 tool output。本地工具使用 PikaCore process 的权限执行。

检查不可信 repository 时应使用 `read-only`。预计会发生变更时，使用默认的 `ask` 并检查每个 proposed side effect。只有在 task、repository、provider 和 local environment 都可信时才应使用 `auto`。

## 已实现控制

### Workspace path

File-tool path 根据 canonical workspace root 解析。Resolver 会拒绝：

- 解析后位于 workspace 外部的 `..` 或 absolute path；
- escape workspace 的现有 symlink；
- 已存在 parent 解析到 workspace 外部的 missing write target；
- 将 repository root 本身作为 file write target。

该边界适用于 path-aware file tool，不会限制 shell tool 创建的 operating-system process。

### 权限与调度

Read-only tool 始终允许。在 `read-only` mode 中，mutating/high-risk tool 会被拒绝。在 `ask` 中，approval callback 会在执行前运行于主线程。在 `auto` 中，side-effecting tool 无需审批即可运行。

Read-only batch 可以并行运行。Write、edit、shell 和 sub-agent call 是串行 barrier。Result 按 model 原始 call order 返回。

### Shell 执行

Shell tool：

- 从 active workspace 启动，按 tool instance 跟踪 cwd；
- 返回真实 process exit code；
- 阻止一小组明确的 destructive pattern；
- 截断过大的 display output，同时保留 head 和 tail；
- 从明确的 cross-platform allowlist 构建 child environment。

Allowlist 包含 `PATH`、locale、temp、home 和 Windows 必需变量等常规 process/runtime variable。名称包含 `KEY`、`TOKEN`、`SECRET`、`PASSWORD` 或 `CREDENTIAL` 的变量会被删除，因此 shell child 不会继承 provider API key。

Command-pattern filter 是 defense in depth，而不是通用 shell policy。等效或经过混淆的命令可能绕过它。`shell=True` 使用平台的常规 shell semantics：Unix-like 系统使用 POSIX shell，Windows 使用 Windows command processor。需要可移植性时，prompt 和 test 不应依赖 POSIX-only syntax。

### 持久化与脱敏

Project state 位于 `.pikacore/`，Git 会忽略该目录。JSON state 使用 atomic replacement；JSONL trace line 以 append 方式写入并 flush。持久化前，PikaCore 会递归脱敏：

- field name 类似 key、token、secret、password 或 credential 的 value；
- 常见 bearer 和 API-key string；
- URL 中嵌入的 credential。

Trace 字符串上限为 4,000 个字符。Session content 不会截断，因此恢复后的 protocol message 在脱敏后仍保持完整。Redaction 是 best-effort，state file 是 plaintext，未加密。

### 恢复

Assistant tool call 会在 side effect 前写入 checkpoint。如果 required checkpoint 无法写入，后续 mutation 会停止。恢复会验证 runtime identity 和 file freshness。保存的 tool call 不会自动重放。`messages` 中没有配对的 call 会收到一个 interrupted result，recovery 会要求在 retry 前检查 workspace。

## Out of scope

PikaCore 当前不提供：

- process、container、VM 或 network isolation；
- 对 arbitrary shell command 的 filesystem confinement；
- encrypted state at rest；
- provider-side data-retention guarantee；
- 完整的 malware 或 destructive-command detector；
- 在用户批准 harmful action 后继续提供保护；
- 覆盖所有 credential format 的 secret scanning。

Sub-agent 继承相同的 workspace、model 和 permission policy。它不能生成另一个 sub-agent，但其 non-read-only work 仍由所选 mode 管理。

## 操作建议

- 从预期的 repository root 运行 PikaCore，并在恢复前检查 `/session`。
- 除非明确接受 unattended mutation，否则保留默认 `ask` mode。
- Resume 或 model switch 后，检查 `/permissions` 和 tool risk classification。
- 将 `.pikacore/`、terminal scrollback 和 `~/.pikacore_history` 视为可能包含敏感信息的 local data。
- 不要提交 `.env`、`.pikacore/`、trace、session 或含有真实 secret 的 benchmark prompt。
- 发生 interrupt 或任何 `incomplete-tool-call` recovery result 后，检查 `git status` 和 `/diff`。

## 报告漏洞

如果仓库启用了 private vulnerability-reporting channel，请使用该渠道。如果没有 private channel，请创建最小化的 GitHub issue，且不要包含 secret、exploit payload 或 private repository content。
