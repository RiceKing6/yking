"""命令行入口：交互 REPL / 单次执行（-p）/ 配置检查（--check）。"""
from __future__ import annotations

import argparse
import json
import sys
import threading

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .agent import Agent, build_system_prompt
from .config import Config, load_config
from .llm import LLMClient
from .memory import ContextCompressor, LongTermMemory
from .orchestrator import READONLY_TOOLS, ROLE_LABELS, Orchestrator, filter_registry
from .plan import STATUS_STYLES, Plan, PlanError
from .planner import PlanExecutor, Planner
from .tools import ToolContext, build_registry

HELP_TEXT = """\
命令:
  /help            显示本帮助
  /clear           清空短期对话历史（长期记忆保留）
  /model           查看当前模型配置
  /plan <需求>     Plan-and-Execute：先拆解成 DAG 任务计划，确认后逐步执行
  /multi <需求>    Multi-Agent：Planner 拆解 → Worker 执行 → Reviewer 验收把关
  /remember 内容   写入长期记忆（跨会话保留，例如 /remember 这个项目用 pytest）
  /memory          查看长期记忆
  /forget 编号     删除某条长期记忆（编号来自 /memory）
  /exit            退出（或按 Ctrl+C / Ctrl+D）

也可以直接输入需求，例如:
  帮我写一个 python 脚本，统计当前目录下各语言的代码行数
  /plan 重构 tools 目录：抽出公共校验逻辑并补上测试
"""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="yking",
        description="yking - 一个专注于写代码/改代码的命令行编码 Agent",
    )
    parser.add_argument("-p", "--prompt", help="单次模式：回答完这一次就退出")
    parser.add_argument("-w", "--workspace", help="Agent 的工作目录（默认为当前目录）")
    parser.add_argument("--model", help="模型名，如 glm-4.6 / deepseek-chat")
    parser.add_argument("--base-url", help="OpenAI 兼容 API 地址")
    parser.add_argument("--api-key", help="API Key（更推荐用环境变量或 .env 文件）")
    parser.add_argument("--max-steps", type=int, help="单轮最多 ReAct 步数（默认 30）")
    parser.add_argument("--auto-approve", action="store_true",
                        help="执行命令、确认计划时不再询问（适合脚本/自动化）")
    parser.add_argument("--plan", action="store_true",
                        help="单次模式（-p）先规划成 DAG 任务再执行")
    parser.add_argument("--multi", action="store_true",
                        help="单次模式（-p）用 Multi-Agent：Planner 拆解 → Worker 执行 → Reviewer 验收")
    parser.add_argument("--no-stream", action="store_true",
                        help="关闭流式输出（整段生成完再显示）")
    parser.add_argument("--parallel", type=int, default=None,
                        help="Plan 模式下并行执行的无依赖任务数（默认 1 = 串行）")
    parser.add_argument("--no-memory", action="store_true",
                        help="禁用长期记忆")
    parser.add_argument("--context-budget", type=int, default=None,
                        help="短期上下文 token 预算，超过自动压缩（默认 40000）")
    parser.add_argument("--check", action="store_true", help="打印当前配置后退出")
    parser.add_argument("--version", action="version", version=f"yking {__version__}")
    return parser.parse_args(argv)


def print_config(console: Console, cfg: Config) -> bool:
    if cfg.api_key:
        key_display = cfg.api_key[:6] + "****" + cfg.api_key[-4:]
    else:
        key_display = "(未设置)"
    console.print(Panel.fit(
        f"api_key      : {key_display}\n"
        f"base_url     : {cfg.base_url}\n"
        f"model        : {cfg.model}\n"
        f"workspace    : {cfg.workspace}\n"
        f"max_steps    : {cfg.max_steps}\n"
        f"auto_approve : {cfg.auto_approve}\n"
        f"stream       : {cfg.stream}\n"
        f"parallel     : {cfg.parallel}\n"
        f"memory       : {cfg.memory}\n"
        f"context_budget: {cfg.context_budget}",
        title=f"yking v{__version__} 配置",
    ))
    return bool(cfg.api_key)


def make_confirm(console: Console):
    """run_command 执行前的 y/N 确认；非交互环境（如管道输入）自动放行。

    加锁保证并行任务的确认请求逐个出现，不会交叉。
    """
    lock = threading.Lock()

    def confirm(command: str) -> bool:
        if not sys.stdin.isatty():
            return True
        with lock:
            console.print(f"[yellow]即将执行命令:[/] {command}")
            try:
                answer = input("  允许执行? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                console.print()
                return False
            return answer in ("y", "yes")
    return confirm


class EventRenderer:
    """把 agent / planner / executor 的事件渲染到终端。

    线程安全：并行执行时任务线程会直接调用本渲染器，所有渲染都在锁内完成。
    流式显示分两种方式：
    - 内联（默认）：delta 原地打印不换行，适合普通对话与串行 Plan；
    - 标签逐行（tagged_tasks=True，并行 Plan 用）：任务内的 delta 以「任务 N」
      前缀整行打印，多个任务的输出交叉出现也不会混淆。
    final 到来时若对应内容已流式输出过，则只收尾，不再重复渲染 Markdown。
    """

    def __init__(self, console: Console, tagged_tasks: bool = False):
        self.console = console
        self.tagged_tasks = tagged_tasks
        self._line_open = False       # 内联模式：流式文本停在行中间
        self._turn_streamed = False   # 内联模式：当前这次模型调用输出过流式文本
        self._streamed_tasks: set[str] = set()  # 标签模式：final 已流式显示过的任务
        self._buffers: dict[str, str] = {}      # 标签模式：任务未成行的流式余量
        self._lock = threading.Lock()

    # -- 标签模式的流式辅助 --------------------------------------------
    def _task_feed(self, task_id: str, delta: str) -> None:
        """把增量并入任务缓冲，遇到完整行就带上任务标签打印。"""
        remainder = self._buffers.get(task_id, "") + delta
        while "\n" in remainder:
            line, _, remainder = remainder.partition("\n")
            if line.strip():
                self.console.print(f"「任务 {task_id}」{line}",
                                   markup=False, highlight=False)
            else:
                self.console.print()
        self._buffers[task_id] = remainder

    def _task_flush(self, task_id: str) -> None:
        """任务流结束时把未成行的余量打印出来。"""
        remainder = self._buffers.pop(task_id, "")
        if remainder.strip():
            self.console.print(f"「任务 {task_id}」{remainder}",
                               markup=False, highlight=False)

    def _break_line(self) -> None:
        if self._line_open:
            self.console.print()
            self._line_open = False

    def _print(self, *args, **kwargs) -> None:
        self._break_line()
        self.console.print(*args, **kwargs)

    def __call__(self, event: dict) -> None:
        with self._lock:
            self._render(event)

    def _render(self, event: dict) -> None:
        kind = event["kind"]
        task_id = event.get("task")
        role = event.get("role")
        if task_id and role:
            task_tag = f"「任务 {task_id}·{ROLE_LABELS.get(role, role)}」"
        elif task_id:
            task_tag = f"「任务 {task_id}」"
        else:
            task_tag = ""

        if kind == "text_delta":
            if self.tagged_tasks and task_id:
                self._task_feed(task_id, event["delta"])
                self._streamed_tasks.add(task_id)
            else:
                self.console.print(event["delta"], end="", markup=False, highlight=False)
                self._line_open = True
                self._turn_streamed = True
            return

        # 标签模式：任务上一段流式文本的余量先落行，再渲染本事件
        if task_id and self._buffers.get(task_id):
            self._task_flush(task_id)

        if kind == "step":
            self._turn_streamed = False

        if kind == "step":
            self._print(f"{task_tag}[dim]—— 第 {event['step']} 步 ——[/dim]")
        elif kind == "text":
            self._print(f"{task_tag}[dim italic]{event['text'].strip()}[/dim italic]")
        elif kind == "tool_call":
            args = event["arguments"]
            try:
                pretty = json.dumps(json.loads(args), ensure_ascii=False) if args.strip() else "{}"
            except json.JSONDecodeError:
                pretty = args
            if len(pretty) > 160:
                pretty = pretty[:160] + "…"
            self._print(f"{task_tag}[cyan]● {event['name']}[/cyan]({pretty})")
        elif kind == "tool_result":
            first_line = next((l for l in event["result"].splitlines() if l.strip()), "")
            if len(first_line) > 160:
                first_line = first_line[:160] + "…"
            style = "red" if event["result"].startswith("错误") else "dim"
            prefix = task_tag + " " if task_tag else "  "
            self._print(f"{prefix}[dim]↳[/dim] [{style}]{first_line}[/{style}]")
        elif kind == "usage":
            prefix = task_tag + " " if task_tag else "  "
            self._print(f"{prefix}[dim]tokens: {event.get('prompt_tokens', '?')} 入 / "
                        f"{event.get('completion_tokens', '?')} 出[/dim]")
        elif kind == "final":
            if task_id and task_id in self._streamed_tasks:
                self._streamed_tasks.discard(task_id)
                self.console.print()
            elif self._line_open or self._turn_streamed:
                self._break_line()
                self.console.print()
            else:
                self.console.print()
                self.console.print(Markdown(event["text"]))
                self.console.print()
        elif kind == "planning":
            self._print("[dim]正在规划…[/dim]")
        elif kind in ("plan_created", "plan_updated"):
            self._print(plan_table(event["plan"]))
        elif kind == "plan_started":
            self._print("[bold]开始执行计划[/bold]\n")
        elif kind == "task_start":
            self._print(f"[bold cyan]▶ 任务 {event['task']}：{event['title']}[/bold cyan]")
        elif kind == "task_done":
            self._print(f"[green]✓ 任务 {event['task']} 完成[/green]\n")
        elif kind == "task_failed":
            self._print(f"[red]✗ 任务 {event['task']} 失败：{event['reason'][:200]}[/red]\n")
        elif kind == "task_skipped":
            self._print(f"[yellow]– 任务 {event['task']} 跳过：{event['reason']}[/yellow]")
        elif kind == "review":
            mark = "✓ 通过" if event.get("verdict") == "pass" else "✗ 打回"
            style = "green" if event.get("verdict") == "pass" else "red"
            self._print(f"[magenta]🔍 任务 {event['task']} 第 {event.get('attempt', '?')} 轮验收："
                        f"[{style}]{mark}[/{style}] [dim]{(event.get('summary') or '')[:150]}[/dim]")
        elif kind == "worker_retry":
            self._print(f"[yellow]↻ 任务 {event['task']} Worker 第 {event.get('attempt', '?')} 次尝试"
                        f"（按审查意见返工）[/yellow] [dim]{(event.get('reason') or '')[:150]}[/dim]")
        elif kind == "compress":
            note = ("早期对话已 LLM 摘要 + 工具输出已裁剪" if event.get("summary")
                    else "早期工具输出已裁剪")
            self._print(f"[yellow]✂ 上下文压缩[/yellow] [dim]"
                        f"~{event.get('before', '?')} → ~{event.get('after', '?')} tokens（{note}）[/dim]")
        elif kind == "memory_saved":
            self._print(f"[green]✓ 已写入长期记忆[/green] [dim]"
                        f"（共 {event.get('count', '?')} 条，跨会话生效）[/dim]")
        elif kind == "replanning":
            self._print(f"[yellow]↻ 任务 {event['task']} 失败，正在重新规划剩余任务…[/yellow]")
        elif kind == "plan_aborted":
            self._print(f"[red]■ 计划中止：{event['reason']}[/red]")
        elif kind == "plan_finished":
            self._print("\n[bold]计划执行完毕[/bold]")
            self.console.print(plan_table(event["plan"]))


def plan_table(plan: Plan) -> Table:
    """把计划渲染成状态表格。"""
    table = Table(title=f"计划：{plan.goal}" if plan.goal else "任务计划")
    table.add_column("ID", style="cyan", justify="center")
    table.add_column("任务")
    if any(t.type != "code" for t in plan.tasks):
        table.add_column("类型", style="dim")
    table.add_column("依赖", style="dim")
    table.add_column("状态")
    for t in plan.tasks:
        icon, color, label = STATUS_STYLES[t.status]
        row = [t.id, t.title]
        if any(x.type != "code" for x in plan.tasks):
            row.append(t.type)
        row += [",".join(t.deps) or "-",
                f"[{color}]{icon} {label}[/{color}]"]
        table.add_row(*row)
    return table


def run_plan_mode(cfg: Config, console: Console, request: str) -> int:
    """Plan-and-Execute：先规划成 DAG 任务，确认后逐步执行。"""
    memory = LongTermMemory(cfg.workspace / ".yking" / "memory.json") if cfg.memory else None
    ctx = ToolContext(
        workspace=cfg.workspace,
        command_timeout=cfg.command_timeout,
        auto_approve=cfg.auto_approve,
        confirm=make_confirm(console),
        memory=memory,
    )
    llm = LLMClient(cfg)
    on_event = EventRenderer(console, tagged_tasks=cfg.parallel > 1)
    planner = Planner(llm, on_event=on_event)
    try:
        plan = planner.create_plan(request)
    except KeyboardInterrupt:
        console.print("\n[yellow]（已取消规划）[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]规划失败: {type(e).__name__}: {e}[/red]")
        return 1

    if sys.stdin.isatty() and not cfg.auto_approve:
        try:
            answer = input("按此计划执行? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]已取消。[/dim]")
            return 130
        if answer in ("n", "no"):
            console.print("[dim]已取消执行。可以继续对话调整需求，或重新 /plan。[/dim]")
            return 0

    executor = PlanExecutor(llm, build_registry(ctx), ctx, cfg, on_event=on_event,
                            memory=memory)
    try:
        plan = executor.run(plan)
    except KeyboardInterrupt:
        console.print("\n[yellow]（已中断执行，已完成任务的成果保留在工作目录）[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]执行出错: {type(e).__name__}: {e}[/red]")
        return 1
    return 1 if any(t.status == "failed" for t in plan.tasks) else 0


def build_agent(cfg: Config, console: Console) -> Agent:
    renderer = EventRenderer(console)
    llm = LLMClient(cfg)
    memory = LongTermMemory(cfg.workspace / ".yking" / "memory.json") if cfg.memory else None
    ctx = ToolContext(
        workspace=cfg.workspace,
        command_timeout=cfg.command_timeout,
        auto_approve=cfg.auto_approve,
        confirm=make_confirm(console),
        memory=memory,
    )
    compressor = ContextCompressor(llm, budget_tokens=cfg.context_budget,
                                   on_event=renderer)
    return Agent(
        llm=llm,
        registry=build_registry(ctx),
        ctx=ctx,
        system_prompt=build_system_prompt(ctx),
        max_steps=cfg.max_steps,
        on_event=renderer,
        compressor=compressor,
        memory=memory,
    )


def run_multi_mode(cfg: Config, console: Console, request: str) -> int:
    """Multi-Agent（主从架构）：Orchestrator + Planner/Worker/Reviewer。"""
    renderer = EventRenderer(console)
    llm = LLMClient(cfg)
    memory = LongTermMemory(cfg.workspace / ".yking" / "memory.json") if cfg.memory else None
    ctx = ToolContext(
        workspace=cfg.workspace,
        command_timeout=cfg.command_timeout,
        auto_approve=cfg.auto_approve,
        confirm=make_confirm(console),
        memory=memory,
    )
    full_registry = build_registry(ctx)
    orchestrator = Orchestrator(
        llm=llm,
        worker_registry=full_registry,
        reviewer_registry=filter_registry(full_registry, READONLY_TOOLS),
        ctx=ctx,
        cfg=cfg,
        memory=memory,
        on_event=renderer,
    )
    try:
        plan = orchestrator.plan(request)
    except KeyboardInterrupt:
        console.print("\n[yellow]（已取消规划）[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]规划失败: {type(e).__name__}: {e}[/red]")
        return 1

    if sys.stdin.isatty() and not cfg.auto_approve:
        try:
            answer = input("按此计划执行（Worker 执行 + Reviewer 验收）? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]已取消。[/dim]")
            return 130
        if answer in ("n", "no"):
            console.print("[dim]已取消执行。可以继续对话调整需求，或重新 /multi。[/dim]")
            return 0

    try:
        plan = orchestrator.execute(plan)
    except KeyboardInterrupt:
        console.print("\n[yellow]（已中断执行，已完成步骤的成果保留在工作目录）[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]执行出错: {type(e).__name__}: {e}[/red]")
        return 1
    return 1 if any(t.status == "failed" for t in plan.tasks) else 0


def run_one_shot(cfg: Config, console: Console, prompt: str) -> int:
    agent = build_agent(cfg, console)
    try:
        agent.run_turn(prompt)
    except KeyboardInterrupt:
        console.print("\n[yellow]（已中断）[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]出错了: {type(e).__name__}: {e}[/red]")
        return 1
    return 0


def run_repl(cfg: Config, console: Console) -> int:
    agent = build_agent(cfg, console)
    console.print(f"[bold cyan]yking[/bold cyan] v{__version__} — 命令行编码 Agent（ReAct + Tool Call）")
    console.print(f"[dim]model={cfg.model}  workspace={cfg.workspace}  输入 /help 查看帮助，/exit 退出[/dim]\n")
    while True:
        try:
            console.print("[bold green]you ❯[/bold green] ", end="")
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n再见！")
            return 0
        if not user_input:
            continue
        if user_input.startswith("/"):
            cmd = user_input.split()[0].lower()
            if cmd in ("/exit", "/quit", "/q"):
                console.print("再见！")
                return 0
            elif cmd == "/help":
                console.print(HELP_TEXT)
            elif cmd == "/clear":
                agent.reset()
                console.print("[dim]已清空短期对话历史（长期记忆保留，/memory 查看）。[/dim]")
            elif cmd == "/model":
                console.print(f"[dim]model={cfg.model}  base_url={cfg.base_url}[/dim]")
            elif cmd == "/plan":
                request = user_input[len("/plan"):].strip()
                if not request:
                    console.print("[dim]用法: /plan <需求>  例如: /plan 写一个批量重命名图片的脚本并测试[/dim]")
                else:
                    run_plan_mode(cfg, console, request)
            elif cmd == "/multi":
                request = user_input[len("/multi"):].strip()
                if not request:
                    console.print("[dim]用法: /multi <需求>  例如: /multi 写一个todo的cli并测试通过[/dim]")
                else:
                    run_multi_mode(cfg, console, request)
            elif cmd == "/remember":
                content = user_input[len("/remember"):].strip()
                if agent.memory is None:
                    console.print("[dim]长期记忆未启用（--no-memory 可查看）。[/dim]")
                elif not content:
                    console.print("[dim]用法: /remember <要记住的内容>[/dim]")
                else:
                    console.print(f"[dim]{agent.memory.save(content)}[/dim]")
            elif cmd == "/memory":
                console.print("[dim]" + (agent.memory.list() if agent.memory
                                         else "长期记忆未启用（--no-memory 可查看）") + "[/dim]")
            elif cmd == "/forget":
                key = user_input[len("/forget"):].strip()
                if agent.memory is None:
                    console.print("[dim]长期记忆未启用。[/dim]")
                elif not key:
                    console.print("[dim]用法: /forget <编号或关键词>  先用 /memory 查看编号[/dim]")
                else:
                    console.print(f"[dim]{agent.memory.forget(key)}[/dim]")
            else:
                console.print(f"[dim]未知命令 {cmd}，输入 /help 查看帮助。[/dim]")
            continue
        try:
            agent.run_turn(user_input)  # 过程与最终回答都由事件回调渲染
        except KeyboardInterrupt:
            console.print("\n[yellow]（已中断本轮，对话历史已回滚）[/yellow]")
        except Exception as e:
            console.print(f"[red]出错了: {type(e).__name__}: {e}[/red]")
            console.print("[dim]提示: 请检查 API key / base_url / model 是否正确、网络是否可用（yking --check）[/dim]")


def main(argv=None) -> int:
    args = parse_args(argv)
    console = Console()
    cfg = load_config(args)

    if args.check:
        return 0 if print_config(console, cfg) else 1

    if not cfg.api_key:
        console.print("[red]未找到 API Key。[/red]")
        console.print("请任选一种方式配置：")
        console.print("  1. 在项目目录创建 [bold].env[/bold] 文件（参考 .env.example）: YKING_API_KEY=sk-xxx")
        console.print("  2. 设置环境变量: [bold]export YKING_API_KEY=sk-xxx[/bold]")
        console.print("  3. 运行时指定: [bold]yking --api-key sk-xxx[/bold]")
        console.print("配置好后用 [bold]yking --check[/bold] 验证。")
        return 1

    if args.prompt:
        if args.multi:
            return run_multi_mode(cfg, console, args.prompt)
        if args.plan:
            return run_plan_mode(cfg, console, args.prompt)
        return run_one_shot(cfg, console, args.prompt)
    return run_repl(cfg, console)


if __name__ == "__main__":
    raise SystemExit(main())
