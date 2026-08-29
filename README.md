[![CI](https://github.com/RiceKing6/yking/actions/workflows/ci.yml/badge.svg)](https://github.com/RiceKing6/yking/actions/workflows/ci.yml)

# yking

一个运行在命令行里的编码 Agent：**写代码、改代码、聊天问答**。

当前版本实现了：**ReAct 循环（思考 → 调用工具 → 观察 → 再思考）+ Tool Call +
Plan-and-Execute（DAG 计划）+ 流式输出 + CLI 交互**。

## 架构

```
┌────────────┐  用户输入/命令   ┌────────────┐  chat(messages, tools) ┌──────────┐
│  cli.py    │ ──────────────► │  agent.py  │ ─────────────────────► │  llm.py  │
│ 交互与渲染  │ ◄────────────── │  ReAct 循环 │ ◄───────────────────── │ OpenAI兼容│
└────────────┘   事件流(渲染)    └─────┬──────┘    统一 dict 响应      └──────────┘
                                      │ 执行工具
                                      ▼
                               ┌────────────┐
                               │  tools/    │ read_file / write_file / edit_file
                               │  工具注册表 │ list_dir / grep_search / run_command
                               └────────────┘
```

- **agent.py** — ReAct 循环核心：用户输入追加进消息历史 → 调用模型 → 若模型返回
  `tool_calls` 就执行工具、把结果以 `role=tool` 消息追加 → 再次调用模型，直到模型给出
  最终回答（或达到最大步数）。异常/中断时回滚消息历史，保证对话状态一致。
- **plan.py** — 计划数据结构与 DAG 校验（拓扑排序、环检测、id 规整）。
- **memory.py** — 记忆系统：长期记忆（跨会话 JSON 落盘、注入 system prompt）+
  上下文压缩器（两级：机械裁剪早期超长工具输出 → LLM 摘要更早对话，切分点只在
  user 消息边界，绝不拆散 tool_calls 与其结果的配对）。
- **planner.py** — Plan-and-Execute 引擎：Planner 让模型通过 `submit_plan` 工具提交/修订
  计划（不合法自动打回重试）；PlanExecutor 按拓扑序执行，每个任务一个独立 ReAct 子 Agent，
  失败自动重规划。
- **orchestrator.py** — Multi-Agent 主从架构：Orchestrator（纯 Python 流程控制）编排
  Planner/Worker/Reviewer 三个子 Agent，Worker 执行、Reviewer 用只读工具实际核验后
  强制 `submit_review` 给 pass/fail 结论，不通过带问题打回重做。
- **llm.py** — OpenAI 兼容接口的薄封装（智谱 GLM / DeepSeek / Qwen / Kimi / OpenAI /
  Ollama 都能用），带限流重试；支持流式输出（内容增量实时回调，工具调用分片自动拼装，
  端点不支持 `stream_options` 时自动降级）；响应统一整理成普通 dict，方便替换和测试。
- **tools/** — 工具注册表。每个工具 = JSON Schema 描述 + Python 函数，
  函数返回的字符串就是模型看到的"观察结果"。
- **cli.py** — 交互 REPL、单次模式（`-p`）、配置检查（`--check`）、命令执行前 y/N 确认。
- **config.py** — 配置优先级：`CLI 参数 > 环境变量 YKING_* > .env 文件 > 默认值`。

## 快速开始

### 1. 克隆仓库并创建独立环境（不污染本地 Python）

```bash
git clone https://github.com/RiceKing6/yking.git
cd yking
conda create -n yking python=3.12 -y
conda activate yking
pip install -e .        # 可编辑安装，改代码立即生效
```

没有 conda 也可以用 venv：`python -m venv .venv`，Windows 激活 `.venv\Scripts\activate`，Linux/macOS 激活 `source .venv/bin/activate`。

### 2. 配置模型 API

复制仓库里的 `.env.example` 为 `.env`，填入你自己的 key（yking 启动时自动读取；`.env` 已被 .gitignore 忽略，不会被提交）：

```bash
# Windows (CMD):  copy .env.example .env
# macOS / Linux:  cp .env.example .env
```

```ini
YKING_API_KEY=sk-你的key
YKING_BASE_URL=https://api.deepseek.com
YKING_MODEL=deepseek-chat
```

或者设置环境变量（Git Bash）：`export YKING_API_KEY=sk-xxx`

任何 OpenAI 兼容服务商都能用，改 `YKING_BASE_URL` + `YKING_MODEL` 即可：

| 服务商 | YKING_BASE_URL | YKING_MODEL |
|---|---|---|
| DeepSeek | `https://api.deepseek.com` | `deepseek-chat` |
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4` | `glm-4.6` |
| Kimi | `https://api.moonshot.cn/v1` | `moonshot-v1-32k` |
| Qwen | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` |
| 本地 Ollama | `http://localhost:11434/v1` | `qwen2.5-coder:7b` |

### 3. 运行

```bash
yking --check                     # 检查配置是否就绪
yking                             # 交互式 REPL
yking -p "用一句话解释什么是 ReAct"   # 单次问答后退出
yking -w D:\some\project          # 指定 Agent 的工作目录
yking --model deepseek-chat --base-url https://api.deepseek.com   # 临时换模型
yking --no-stream                 # 关闭流式输出（默认边生成边显示，整段渲染 Markdown）
```

REPL 内置命令：`/help`、`/clear`（清空对话历史）、`/model`、`/exit`。

## 内置工具

| 工具 | 说明 |
|---|---|
| `read_file` | 带行号读取文件，支持 `offset` / `limit` 分段 |
| `write_file` | 整体写入文件（覆盖），自动创建父目录 |
| `edit_file` | 精确字符串替换，要求 `old_string` 唯一匹配 |
| `list_dir` | 列出目录一层内容 |
| `grep_search` | 正则搜索文件内容，返回 `文件:行号:内容` |
| `run_command` | 执行 shell 命令并返回 stdout/stderr（默认执行前 y/N 确认；`--auto-approve` 或 `YKING_AUTO_APPROVE=1` 跳过） |

## Plan-and-Execute 模式（DAG）

复杂任务可以先规划再执行：

```bash
yking --plan -p "做一个XXX"     # 单次模式：规划 → 确认 → 执行
```

REPL 里用 `/plan <需求>`，例如 `/plan 用python写一个批量重命名图片的脚本并测试`。

工作流程：

1. **规划**：Planner 调用 LLM 把目标拆解成任务列表（每个任务有 `id / title / goal / deps`），
   必须通过 `submit_plan` 工具提交；框架校验 id 唯一、依赖存在、无环（DAG），并算出拓扑执行序，
   不合法会带着错误信息打回给模型重试（最多 3 次）。
2. **确认**：展示计划表格，交互模式下询问 `按此计划执行? [Y/n]`（`--auto-approve` 时跳过）。
3. **执行**：按 DAG 依赖关系调度；每个任务是一个独立的 ReAct 子 Agent（可用全部工具），
   依赖任务的产出摘要会注入后续任务的上下文。加 `--parallel N`（如 `--parallel 3`）可让
   无依赖的任务并行执行，输出按「任务 N」标签逐行区分；默认 1 = 串行（内联流式显示）。
4. **失败处理**：任务失败（执行异常，或最终回答以 `[FAILED]:` 开头）时自动**重规划**剩余任务
   （最多 2 次，新任务 id 与已完成任务冲突时自动重命名为 R1、R2…）；重规划器认为无法继续时
   中止整个计划。
5. **展示**：实时显示每个任务内的工具调用与 token 用量，结束输出状态总表。

注意：并行只是调度方式，任务之间如果会写同一批文件，应在计划里用 deps 表达成先后依赖来避免冲突。

## Multi-Agent 模式（Orchestrator + Planner/Worker/Reviewer）

主从架构的多角色协作，比 Plan-and-Execute 多一道**独立验收**：

```bash
yking --multi -p "创建 calc.py 实现 add/sub，写测试并跑通"   # 单次模式
```

REPL 里用 `/multi <需求>`。

- **Orchestrator（主）**：纯 Python 流程控制，不做 LLM 调用——分发任务、控制流程：
  规划 → 按依赖逐步分发 → Worker/Reviewer 循环 → 汇总。
- **Planner（规划者）**：把需求拆成步骤列表，每步标注 `type`（research 只读调研 /
  code 写改文件 / command 执行命令）、任务书、**可核验的验收标准**、依赖关系。
- **Worker（执行者）**：拿到步骤任务书后用工具干活；research 类型只发只读工具（最小权限）。
- **Reviewer（检查者）**：不轻信 Worker 自述，用**只读工具实际核验**（读文件、跑命令），
  再通过 `submit_review` 强制给出 `pass/fail` + 结构化结论；fail 必须附具体问题清单，
  Worker 带着问题返工（每步最多 3 次尝试，仍不通过则该步骤失败、依赖步骤级联跳过）。
- Reviewer 自身故障时保守放行并在报告中标注"未经审查"（fail-open），不阻塞整个计划。

与 Plan-and-Execute 的区别：/plan 的执行器自查自报；/multi 多一个独立验收角色，
质量把关更强，代价是每步多 1~2 次 LLM 调用。

## 记忆系统（短期 + 长期 + 上下文压缩）

- **短期记忆**：当前会话的对话历史。有 token 预算（默认 40000，`--context-budget` 调整），
  逼近预算（80%）时自动触发压缩，终端会显示 `✂ 上下文压缩` 与前后 token 估算。
- **上下文压缩**：两级策略——先机械裁剪早期消息里的超长工具输出（消息结构与配对原样
  保留，不花 LLM 调用）；仍超限则把更早的一段对话用 LLM 摘要成一条消息替换。切分点只选
  在 user 消息边界，绝不拆散 assistant(tool_calls) 与其 tool 结果。
- **长期记忆**：跨会话保留，落盘在 `<工作目录>/.yking/memory.json`：
  - 模型可通过 `save_memory` 工具保存用户偏好/项目约定（仅在明确稳定的偏好时使用），
    全部记忆会注入 system prompt，跨会话生效；
  - 用户命令：`/remember 内容` 写入、`/memory` 查看、`/forget 编号|关键词` 删除；
  - 上限 50 条（自动淘汰最旧）、单条 500 字符；`--no-memory` 关闭。
另外在 `conda run` 等包装器下 stdin 可能被误判为终端，自动化场景请始终加 `--auto-approve`。

## 如何新增一个工具

1. 在 `yking/tools/` 里写函数，签名统一为 `(ctx: ToolContext, **参数) -> str`，
   返回的字符串就是给模型看的观察结果（错误也用字符串返回，让模型有机会自我修正）；
2. 在 `yking/tools/__init__.py` 的 `build_registry()` 里注册名字、描述和 JSON Schema。

## 测试

不需要 API Key 的冒烟测试（工具实现 + ReAct 循环 + 异常回滚 + DAG/规划/执行）：

```bash
conda run -n yking python tests/test_smoke.py       # 基础 ReAct 循环
conda run -n yking python tests/test_plan.py        # Plan-and-Execute
conda run -n yking python tests/test_parallel.py    # DAG 并行执行
conda run -n yking python tests/test_stream.py      # 流式输出拼装/降级
conda run -n yking python tests/test_memory.py      # 记忆系统/上下文压缩
conda run -n yking python tests/test_multi_agent.py # Multi-Agent 三角色协作
```

## 已知限制

- 工具可以读写工作目录之外的绝对路径（本地个人工具的取舍，未做路径白名单）；
- `run_command` 先完整缓冲命令输出再截断（极端大输出会占用内存）；
- 若系统设置过 `OPENAI_API_KEY` 而未设置 `YKING_API_KEY`，会拿它请求默认的智谱地址导致 401，请在 `.env` 显式配置；
- 上下文压缩的机械级（找不到轮次边界时）会把较早的用户消息截断到 600 字符，LLM 摘要级则完整保留；
- 无法按 UTF-8 或系统编码解码的文件（二进制/旧编码如 GBK）会被 `edit_file`/`write_file` 拒绝处理，防止内容损坏，请先转码；
- 流式输出时中间思考与最终回答样式相同（生成时无法预知后续是否调用工具）。
