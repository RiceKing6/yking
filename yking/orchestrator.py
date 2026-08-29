"""Multi-Agent（主从架构）：Orchestrator 编排器 + Planner/Worker/Reviewer 三角色子 Agent。

- Orchestrator（主）：纯 Python 流程控制，不做 LLM 调用。负责任务分发与流程控制：
  先让 Planner 把需求拆成带类型与依赖的步骤（DAG），再按拓扑序逐步分发——
  Worker 执行 → Reviewer 验收 → 不通过则带着问题打回重做（最多 MAX_WORKER_ATTEMPTS 次）；
  前置步骤失败/跳过的步骤级联跳过，全部走完后输出最终状态。
- Planner（规划者）：把需求拆成步骤列表，每步标注 type（research/code/command）、
  任务书 goal、验收标准 acceptance、依赖 deps。复用 planner.Planner 的
  "强制工具提交 + 校验打回重试" 机制，只是换成角色提示词与带 type/acceptance 的 Schema。
- Worker（执行者）：按步骤指令调用工具干活（research 类型只给只读工具），产出执行报告。
- Reviewer（检查者）：不轻信 Worker 自述，用只读工具实际核验（读文件/跑命令），
  然后通过 submit_review 强制给出 pass/fail 结论；fail 必须附带具体问题清单供返工。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from .agent import Agent, build_system_prompt
from .config import Config
from .plan import (MAX_PLAN_TASKS, STEP_TYPES, Plan, PlanTask, parse_plan,
                   validate_dag)
from .planner import Planner
from .tools import ToolContext, execute_tool

EventCallback = Callable[[dict], None]

MAX_WORKER_ATTEMPTS = 3   # 每个步骤 Worker 最多尝试次数（首次 + 2 次按审查意见返工）
REVIEWER_MAX_ROUNDS = 10  # Reviewer 单次验收的 LLM 轮数上限（含核验用的工具调用）

ROLE_LABELS = {"worker": "Worker", "reviewer": "Reviewer", "planner": "Planner"}

# Reviewer 的工具白名单：验收者只读不写（最小权限），也不给它 save_memory
READONLY_TOOLS = ("read_file", "list_dir", "grep_search", "run_command")

# Planner 角色的提交 Schema：比普通 /plan 多 type 与 acceptance（Reviewer 的判据）
SUBMIT_STEPS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_plan",
        "description": "提交步骤计划。必须通过调用本工具提交。",
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "总体目标的一句话概括"},
                "tasks": {
                    "type": "array",
                    "description": "步骤列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "步骤编号，如 \"1\""},
                            "title": {"type": "string", "description": "一句话标题"},
                            "type": {"type": "string", "enum": list(STEP_TYPES),
                                     "description": ("research=只读调研（读代码/查资料）；"
                                                     "code=编写或修改文件；"
                                                     "command=执行命令完成操作或验证")},
                            "goal": {"type": "string",
                                     "description": "写给 Worker 的任务书：具体做什么、涉及哪些文件/命令"},
                            "acceptance": {"type": "string",
                                           "description": ("写给 Reviewer 的验收标准，必须可核验"
                                                           "（如：xx 文件存在且包含 xx；命令 xx 输出 xx）")},
                            "deps": {"type": "array", "items": {"type": "string"},
                                     "description": "依赖的前置步骤 id；无依赖为空数组"},
                        },
                        "required": ["id", "title", "type", "goal", "acceptance", "deps"],
                    },
                },
            },
            "required": ["goal", "tasks"],
        },
    },
}

# Reviewer 的结论工具：强制结构化，避免解析自由文本
SUBMIT_REVIEW_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_review",
        "description": "提交验收结论。必须在实际核验工作成果后调用本工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["pass", "fail"],
                            "description": "pass=验收标准全部满足；fail=存在不满足项"},
                "summary": {"type": "string",
                            "description": "一段话结论：核验了什么、结果如何"},
                "issues": {"type": "array", "items": {"type": "string"},
                           "description": ("verdict=fail 时必填：具体问题清单（哪个文件、什么现象、"
                                           "期望 vs 实际），Worker 会逐条返工；pass 时为空数组")},
            },
            "required": ["verdict", "summary", "issues"],
        },
    },
}

PLANNER_ROLE_SYSTEM = f"""你是 yking Multi-Agent 团队的规划者（Planner）。用户会给你一个目标，你的唯一职责是把目标拆解成一份可执行的步骤计划，并通过调用 submit_plan 工具提交。

拆解要求：
1. 每个步骤是一个可独立执行、可独立验收的工作单元；通常 2~{MAX_PLAN_TASKS} 步，目标很简单时 1 步也可以。
2. 每步必须标注 type：
   - research：只读调研（读代码、查资料、理清现状），不产生任何修改；
   - code：编写或修改文件，是产出的主要来源；
   - command：执行命令完成操作或验证（跑测试、装依赖等）。
3. goal 是写给 Worker 的任务书：具体做什么、涉及哪些文件/命令，要具体到可以直接动手。
4. acceptance 是写给 Reviewer 的验收标准：必须可核验（"a.txt 存在且内容为 X"、"pytest 全部通过"），Reviewer 会逐条对照。
5. 用 deps 表达依赖（DAG）：有产出依赖或会写同一批文件的步骤必须串行；严禁循环依赖。
6. 计划必须完整覆盖目标。你只规划，不执行任何步骤。
"""


@dataclass
class ReviewVerdict:
    """Reviewer 的结构化验收结论。"""
    verdict: str  # "pass" / "fail"
    summary: str = ""
    issues: list[str] = field(default_factory=list)

    @classmethod
    def from_args(cls, args) -> "ReviewVerdict":
        if not isinstance(args, dict):
            raise ValueError("submit_review 参数必须是 JSON 对象")
        verdict = str(args.get("verdict") or "").strip().lower()
        if verdict not in ("pass", "fail"):
            raise ValueError(f"verdict 必须是 pass 或 fail，得到: {verdict!r}")
        issues_raw = args.get("issues") or []
        if not isinstance(issues_raw, list):
            raise ValueError("issues 必须是字符串数组")
        issues = [str(x).strip() for x in issues_raw if str(x).strip()]
        if verdict == "fail" and not issues:
            raise ValueError("verdict=fail 时必须给出具体 issues（Worker 要靠它们返工）")
        summary = str(args.get("summary") or "").strip()
        return cls(verdict=verdict, summary=summary, issues=issues)


def filter_registry(registry: dict, names) -> dict:
    """按名字过滤工具注册表（最小权限：按角色发工具）。"""
    return {k: v for k, v in registry.items() if k in set(names)}


class Orchestrator:
    """主从架构的编排器：任务分发与流程控制（自身不做 LLM 调用）。"""

    def __init__(self, llm, worker_registry: dict, reviewer_registry: dict,
                 ctx: ToolContext, cfg: Config, memory=None,
                 on_event: EventCallback | None = None):
        self.llm = llm
        self.worker_registry = worker_registry
        self.reviewer_registry = reviewer_registry
        self.ctx = ctx
        self.cfg = cfg
        self.memory = memory
        self.on_event = on_event
        self.planner = Planner(llm, on_event=on_event,
                               system_prompt=PLANNER_ROLE_SYSTEM,
                               schema=SUBMIT_STEPS_SCHEMA)
        self._worker_schemas = [t.to_openai_schema() for t in self.worker_registry.values()]
        self._reviewer_schemas = ([t.to_openai_schema() for t in self.reviewer_registry.values()]
                                  + [SUBMIT_REVIEW_SCHEMA])

    # ------------------------------------------------------------------
    def _emit(self, event: dict) -> None:
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:
                pass

    def _wrap(self, task_id: str, role: str) -> Callable[[dict], None]:
        """给子 Agent 的事件打上任务与角色标签。"""
        def on_event(event: dict) -> None:
            self._emit({**event, "task": task_id, "role": role})
        return on_event

    # -- 阶段一：Planner 拆解步骤 --------------------------------------
    def plan(self, request: str) -> Plan:
        return self.planner.create_plan(request)

    # -- 阶段二：按 DAG 分发 Worker → Reviewer 循环 ---------------------
    def execute(self, plan: Plan) -> Plan:
        order = validate_dag(plan)
        self._emit({"kind": "plan_started", "plan": plan})
        for tid in order:
            step = plan.task(tid)
            dep_tasks = [plan.task(d) for d in step.deps]
            blocked = [d.id for d in dep_tasks if d is not None
                       and d.status in ("failed", "skipped")]
            if blocked:
                step.status = "skipped"
                step.report = "前置步骤未完成: " + ", ".join(blocked)
                self._emit({"kind": "task_skipped", "task": step.id, "reason": step.report})
                continue
            self._run_step(plan, step)
        self._emit({"kind": "plan_finished", "plan": plan})
        return plan

    # ------------------------------------------------------------------
    def _run_step(self, plan: Plan, step: PlanTask) -> None:
        """单个步骤的主循环：Worker 干活 → Reviewer 验收 → 不通过打回重做。"""
        step.status = "running"
        self._emit({"kind": "task_start", "task": step.id, "title": step.title})
        dep_reports = []
        for d in step.deps:
            dt = plan.task(d)
            if dt is not None and dt.report:
                dep_reports.append(f"## 前置步骤 {dt.id}. {dt.title} 的产出\n{dt.report}")

        issues: list[str] = []
        for attempt in range(1, MAX_WORKER_ATTEMPTS + 1):
            try:
                worker_report = self._run_worker(step, attempt, issues, dep_reports)
            except KeyboardInterrupt:
                step.status = "failed"
                step.report = "被用户中断"
                self._emit({"kind": "task_failed", "task": step.id, "reason": step.report})
                raise
            except Exception as e:
                step.status = "failed"
                step.report = f"Worker 执行异常: {type(e).__name__}: {e}"
                self._emit({"kind": "task_failed", "task": step.id, "reason": step.report})
                return

            review = None
            if worker_report.upper().startswith("[FAILED]"):
                # Worker 自报失败：没有成果可验收，不打扰 Reviewer，直接带原因返工
                issues = [worker_report[len("[FAILED]"):].lstrip(" :：") or "Worker 自报失败"]
            else:
                try:
                    review = self._review(step, worker_report)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    # Reviewer 自身故障：保守放行，但明确标注未经审查
                    step.status = "done"
                    step.report = (f"Worker 报告: {worker_report[:1500]}\n"
                                   f"[未经审查：Reviewer 异常 {type(e).__name__}: {e}]")
                    self._emit({"kind": "task_done", "task": step.id})
                    return
                self._emit({"kind": "review", "task": step.id, "attempt": attempt,
                            "verdict": review.verdict,
                            "summary": review.summary or "; ".join(review.issues)})
                if review.verdict == "pass":
                    step.status = "done"
                    step.report = (f"Worker 报告: {worker_report[:1500]}\n"
                                   f"Reviewer 结论: {review.summary[:500]}")
                    self._emit({"kind": "task_done", "task": step.id})
                    return
                issues = review.issues

            if attempt < MAX_WORKER_ATTEMPTS:
                self._emit({"kind": "worker_retry", "task": step.id, "attempt": attempt + 1,
                            "reason": "; ".join(issues)[:200]})

        step.status = "failed"
        step.report = (f"连续 {MAX_WORKER_ATTEMPTS} 次未通过验收。最后一次问题: "
                       + "; ".join(issues))[:500]
        self._emit({"kind": "task_failed", "task": step.id, "reason": step.report})

    # -- Worker --------------------------------------------------------
    def _run_worker(self, step: PlanTask, attempt: int, issues: list[str],
                    dep_reports: list[str]) -> str:
        system = self._worker_system(step)
        parts = [f"请完成步骤 [{step.id}] {step.title}",
                 f"\n任务书：\n{step.goal}",
                 f"\n验收标准（Reviewer 将逐条核验）：\n{step.acceptance or '（见任务书）'}"]
        if dep_reports:
            parts.append("\n前置步骤产出（供参考）：\n" + "\n".join(dep_reports))
        if attempt > 1 and issues:
            parts.append("\n# 上一轮验收未通过，Reviewer 指出的问题（必须逐条解决）：\n"
                         + "\n".join(f"- {i}" for i in issues))
        # research 类型只给只读工具（最小权限）
        registry = (self.reviewer_registry if step.type == "research"
                    else self.worker_registry)
        agent = Agent(
            self.llm,
            registry,
            self.ctx,
            system_prompt=system,
            max_steps=self.cfg.max_steps,
            on_event=self._wrap(step.id, "worker"),
            memory=self.memory,
        )
        return (agent.run_turn("\n".join(parts)) or "").strip()

    def _worker_system(self, step: PlanTask) -> str:
        type_hint = {
            "research": "本步骤类型为 research：只读调研，不要创建或修改任何文件。",
            "code": "本步骤类型为 code：编写或修改代码/文件，以验收标准为目标。",
            "command": "本步骤类型为 command：执行命令完成操作或验证，并核对输出。",
        }[step.type]
        return (build_system_prompt(self.ctx)
                + "\n\n# 你的角色：Worker（执行者）\n"
                "你是 Multi-Agent 团队中的执行者，只负责完成编排器分配给你的这一个步骤，"
                "不要越界做其他步骤的事。\n"
                f"- {type_hint}\n"
                "- 完成后在最终回答里给出执行报告：做了什么、关键文件/命令、"
                "与验收标准的对照结果。\n"
                "- 确实无法完成时，最终回答以 [FAILED]: 开头并说明卡在哪里。")

    # -- Reviewer ------------------------------------------------------
    def _review(self, step: PlanTask, worker_report: str) -> ReviewVerdict:
        """Reviewer 子 Agent：可用只读工具实际核验，最后强制 submit_review 给结论。"""
        messages: list[dict] = [
            {"role": "system", "content": self._reviewer_system()},
            {"role": "user", "content": "\n".join([
                f"请验收步骤 [{step.id}] {step.title}。",
                f"\n任务书：\n{step.goal}",
                f"\n验收标准（逐条核验）：\n{step.acceptance or '（见任务书）'}",
                f"\nWorker 执行报告：\n{worker_report[:3000]}",
                "\n请实际核验工作成果（读文件/跑命令），不要轻信报告，"
                "然后调用 submit_review 给出结论。",
            ])},
        ]
        for _ in range(REVIEWER_MAX_ROUNDS):
            reply = self.llm.chat(
                messages, self._reviewer_schemas,
                on_text_delta=lambda d: self._emit({"kind": "text_delta", "delta": d,
                                                    "task": step.id, "role": "reviewer"}))
            content = reply.get("content") or ""
            tool_calls = reply.get("tool_calls") or []
            assistant_msg: dict = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in tool_calls]
            messages.append(assistant_msg)

            review_call = next((tc for tc in tool_calls
                                if tc["name"] == "submit_review"), None)
            if review_call is None:
                if not tool_calls:
                    messages.append({"role": "user",
                                     "content": "请通过调用 submit_review 工具提交验收结论。"})
                    continue
                # 核验用的普通工具调用：执行并把结果喂回去
                for tc in tool_calls:
                    self._emit({"kind": "tool_call", "task": step.id, "role": "reviewer",
                                "name": tc["name"], "arguments": tc["arguments"]})
                    try:
                        args = json.loads(tc["arguments"] or "{}")
                    except json.JSONDecodeError:
                        result = f"错误: 参数不是合法 JSON: {tc['arguments'][:100]}"
                    else:
                        result = execute_tool(self.reviewer_registry, self.ctx,
                                              tc["name"],
                                              args if isinstance(args, dict) else {})
                    self._emit({"kind": "tool_result", "task": step.id,
                                "role": "reviewer", "result": result})
                    messages.append({"role": "tool", "tool_call_id": tc["id"],
                                     "content": result})
                continue

            for tc in tool_calls:
                if tc["id"] != review_call["id"]:
                    messages.append({"role": "tool", "tool_call_id": tc["id"],
                                     "content": "忽略：本轮只接受 submit_review。"})
            try:
                args = json.loads(review_call["arguments"] or "{}")
                verdict = ReviewVerdict.from_args(args)
            except (json.JSONDecodeError, ValueError) as e:
                messages.append({"role": "tool", "tool_call_id": review_call["id"],
                                 "content": f"结论不合法: {e}。请重新调用 submit_review。"})
                continue
            messages.append({"role": "tool", "tool_call_id": review_call["id"],
                             "content": "OK: 验收结论已记录。"})
            return verdict
        raise RuntimeError(f"Reviewer 在 {REVIEWER_MAX_ROUNDS} 轮内未给出有效结论")

    def _reviewer_system(self) -> str:
        return (build_system_prompt(self.ctx)
                + "\n\n# 你的角色：Reviewer（检查者）\n"
                "你是 Multi-Agent 团队中的验收者，职责是判断 Worker 的工作成果是否达标：\n"
                "1. 不要轻信 Worker 的自述，必须实际核验：读它声称修改的文件、"
                "运行它声称通过的命令；\n"
                "2. 对照验收标准逐条检查，全部满足才 pass；不追求完美，但标准必须守住；\n"
                "3. 你只验收，不亲自修改代码；\n"
                "4. 检查完后调用 submit_review 提交结论：fail 时 issues 必须具体可执行"
                "（哪个文件、什么现象、期望 vs 实际），Worker 会根据它们逐条返工。")
