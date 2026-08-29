"""Plan-and-Execute 引擎：

- Planner：让 LLM 把目标拆解成任务 DAG，必须通过 submit_plan 工具提交，
  框架负责解析、DAG 校验，不合法就打回重试。
- PlanExecutor：按 DAG 依赖关系调度执行。默认串行（拓扑序）；--parallel N 时
  无依赖的任务在线程中并行执行。每个任务交给一个独立的 ReAct 子 Agent（可用全部工具），
  依赖任务的产出摘要会注入后续任务的上下文；任务失败时触发重规划（最多 max_replans 次）。
"""
from __future__ import annotations

import json
import queue
import threading
from typing import Callable

from .agent import Agent, build_system_prompt
from .config import Config
from .plan import (MAX_PLAN_TASKS, STATUS_STYLES, Plan, PlanError, PlanTask,
                   parse_plan, validate_dag)
from .tools import ToolContext

EventCallback = Callable[[dict], None]

# 规划器唯一可用的工具：用"强制工具调用"的方式拿到结构化计划，比解析自由文本可靠
SUBMIT_PLAN_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_plan",
        "description": "提交（或重新提交）任务计划。规划结果必须通过调用本工具提交。",
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "总体目标的一句话概括"},
                "tasks": {
                    "type": "array",
                    "description": "任务列表；重规划时传空数组表示放弃并说明原因",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "任务编号，如 \"1\""},
                            "title": {"type": "string", "description": "一句话标题"},
                            "goal": {"type": "string",
                                     "description": "具体要做什么、如何验收（写给执行者的任务书）"},
                            "deps": {"type": "array", "items": {"type": "string"},
                                     "description": "依赖的前置任务 id 列表；无依赖为空数组"},
                        },
                        "required": ["id", "title", "goal", "deps"],
                    },
                },
                "abort_reason": {"type": "string",
                                 "description": "仅当 tasks 为空时填写：为什么放弃"},
            },
            "required": ["goal", "tasks"],
        },
    },
}

PLANNER_SYSTEM = f"""你是 yking 的规划器（Planner）。用户会给你一个目标，你的唯一职责是把目标拆解成一份可执行的任务计划，并通过调用 submit_plan 工具提交。

拆解要求：
1. 每个任务应是一个可独立执行、可独立验证的工作单元（例如"实现某模块"、"编写某脚本"、"运行测试并修复问题"）；通常 2~{MAX_PLAN_TASKS} 个任务，目标很简单时拆 1 个也可以。
2. 任务 id 用 "1"、"2"、"3"… 按逻辑顺序编号。
3. goal 是写给执行者的任务书：具体做什么、涉及哪些文件/命令、怎样算完成，要具体到可以直接动手。
4. 用 deps 表达任务间依赖（DAG）：只有确实需要前序任务产出的才写依赖；没有依赖的任务 deps 为空数组；严禁循环依赖。
5. 会写同一批文件、操作同一资源的任务必须用 deps 串成先后关系，避免并发冲突。
6. 计划必须完整覆盖目标。你只负责规划，不要执行任何任务。
"""

REPLAN_SYSTEM = """你是 yking 的重规划器。之前的计划在执行中出现了失败，你要提交一份"剩余工作"的修订计划（通过调用 submit_plan 工具）。

要求：
1. 只规划还没完成的工作：可以新增、删除、修改剩余任务；已完成和已失败的任务不要再列进来。
2. 任务 id 从 "1" 重新编号；deps 只能引用本份新计划内的任务 id。
3. 如果失败的部分可以绕过或与剩余工作无关，直接规划剩余部分即可。
4. 如果判断目标已经无法达成，tasks 传空数组，并在 abort_reason 里说明原因。
"""


class Planner:
    """用 LLM 生成 / 修订计划。

    system_prompt / schema 可替换：Multi-Agent 的 Planner 角色复用同一套
    "强制工具提交 + 校验打回重试" 机制，只是提示词和 Schema 不同。
    """

    def __init__(self, llm, on_event: EventCallback | None = None, max_attempts: int = 3,
                 system_prompt: str | None = None, schema: dict | None = None):
        self.llm = llm
        self.on_event = on_event
        self.max_attempts = max_attempts
        self.system_prompt = system_prompt or PLANNER_SYSTEM
        self.schema = schema or SUBMIT_PLAN_SCHEMA

    def _emit(self, event: dict) -> None:
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:
                pass

    def _chat_until_plan(self, system: str, user: str, allow_empty: bool = False):
        """驱动模型直到提交合法计划。返回 (Plan, abort_reason)。

        allow_empty=True 用于重规划：允许提交空任务列表表示放弃。
        """
        messages: list[dict] = [{"role": "system", "content": system},
                                {"role": "user", "content": user}]
        last_text = ""
        for _ in range(self.max_attempts):
            reply = self.llm.chat(
                messages, [self.schema],
                on_text_delta=lambda d: self._emit({"kind": "text_delta", "delta": d}))
            content = reply.get("content") or ""
            tool_calls = reply.get("tool_calls") or []
            assistant_msg: dict = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in tool_calls
                ]
            messages.append(assistant_msg)
            if content:
                last_text = content

            plan_call = next((tc for tc in tool_calls if tc["name"] == "submit_plan"), None)
            if plan_call is None:
                for tc in tool_calls:
                    messages.append({"role": "tool", "tool_call_id": tc["id"],
                                     "content": "规划阶段只接受 submit_plan，请不要调用其他工具。"})
                if not tool_calls:
                    messages.append({"role": "user",
                                     "content": "请立即通过调用 submit_plan 工具提交计划，不要用纯文字回答。"})
                continue
            for tc in tool_calls:
                if tc["id"] != plan_call["id"]:
                    messages.append({"role": "tool", "tool_call_id": tc["id"],
                                     "content": "忽略：规划阶段只接受 submit_plan。"})
            try:
                args = json.loads(plan_call["arguments"] or "{}")
                if not isinstance(args, dict):
                    raise PlanError("submit_plan 参数必须是 JSON 对象")
                plan = parse_plan(args)
            except (json.JSONDecodeError, PlanError) as e:
                messages.append({"role": "tool", "tool_call_id": plan_call["id"],
                                 "content": f"计划不合法: {e}。请修正后重新调用 submit_plan。"})
                continue
            if not plan.tasks and not allow_empty:
                messages.append({"role": "tool", "tool_call_id": plan_call["id"],
                                 "content": "计划不能为空，请至少拆解出 1 个任务。"})
                continue
            messages.append({"role": "tool", "tool_call_id": plan_call["id"],
                             "content": "OK: 计划已接受。"})
            abort_reason = "" if plan.tasks else str(args.get("abort_reason") or "").strip()
            return plan, abort_reason
        raise PlanError(
            f"规划器连续 {self.max_attempts} 次未能提交合法计划。最后一次输出：{last_text[:300]}")

    def create_plan(self, request: str) -> Plan:
        self._emit({"kind": "planning"})
        plan, _ = self._chat_until_plan(
            self.system_prompt,
            f"目标：\n{request}\n\n请把目标拆解成任务计划，并用 submit_plan 提交。",
            allow_empty=False,
        )
        self._emit({"kind": "plan_created", "plan": plan})
        return plan

    def replan(self, plan: Plan, failed: PlanTask):
        """针对失败任务修订剩余计划。返回 (新计划|None, abort_reason)。"""
        self._emit({"kind": "replanning", "task": failed.id})
        lines = [f"总体目标：{plan.goal}", "", "当前进度："]
        for t in plan.tasks:
            line = f"- [{t.status}] {t.id}. {t.title}"
            if t.status == "failed":
                line += f"（失败原因：{t.report[:200]}）"
            elif t.status == "done" and t.report:
                line += f"（产出摘要：{t.report[:150]}）"
            lines.append(line)
        lines.append(
            f"\n任务 {failed.id} 执行失败。请提交剩余任务的修订计划"
            "（id 从 1 重新编号，deps 只能引用新计划内的任务）；"
            "若认为目标已无法达成，tasks 传空数组并在 abort_reason 说明。")
        new_plan, abort_reason = self._chat_until_plan(
            REPLAN_SYSTEM, "\n".join(lines), allow_empty=True)
        if not new_plan.tasks:
            return None, abort_reason
        return new_plan, ""


class PlanExecutor:
    """按 DAG 依赖关系调度执行：默认串行（拓扑序），--parallel N 时无依赖任务并行跑。

    每个任务一个独立 ReAct 子 Agent；失败触发重规划。
    """

    def __init__(self, llm, registry, ctx: ToolContext, cfg: Config,
                 on_event: EventCallback | None = None, max_replans: int = 2,
                 memory=None):
        self.llm = llm
        self.registry = registry
        self.ctx = ctx
        self.cfg = cfg
        self.on_event = on_event
        self.max_replans = max_replans
        self.memory = memory  # LongTermMemory；任务执行器也能读写同一份长期记忆
        self.max_workers = max(1, int(getattr(cfg, "parallel", 1) or 1))
        self._cancelled = False   # Ctrl+C 后置位，正在跑的任务收尾时不再写回状态
        self.planner = Planner(llm, on_event=on_event)

    def _emit(self, event: dict) -> None:
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:
                pass

    def run(self, plan: Plan) -> Plan:
        try:
            order = validate_dag(plan)
        except PlanError as e:
            self._emit({"kind": "plan_aborted", "reason": str(e)})
            return plan
        self._emit({"kind": "plan_started", "plan": plan})

        done_q: queue.Queue = queue.Queue()
        inflight: set[str] = set()

        def worker(task: PlanTask) -> None:
            try:
                self._run_task(plan, task)
            except BaseException as e:  # 防御：保证线程一定收尾并通知主循环
                if task.status != "failed":
                    task.status = "failed"
                    task.report = f"线程异常: {type(e).__name__}: {e}"
                    self._emit({"kind": "task_failed", "task": task.id,
                                "reason": task.report})
            finally:
                done_q.put(task.id)

        replans_left = self.max_replans
        try:
            while True:
                # 按拓扑序把并行窗口填满：依赖全部完成的任务立即开跑
                for tid in order:
                    if len(inflight) >= self.max_workers:
                        break
                    task = plan.task(tid)
                    if task is None or task.status != "pending":
                        continue
                    dep_tasks = [plan.task(d) for d in task.deps]
                    bad = [d for d in dep_tasks
                           if d is not None and d.status in ("failed", "skipped")]
                    if bad:
                        task.status = "skipped"
                        task.report = "前置任务未完成: " + ", ".join(d.id for d in bad)
                        self._emit({"kind": "task_skipped", "task": task.id,
                                    "reason": task.report})
                        continue
                    if any(d is None or d.status != "done" for d in dep_tasks):
                        continue  # 依赖还没全部完成（可能正在跑），等下一轮调度
                    inflight.add(task.id)
                    threading.Thread(target=worker, args=(task,), daemon=True,
                                     name=f"yking-task-{task.id}").start()

                if not inflight:
                    if all(t.status in ("done", "skipped", "failed") for t in plan.tasks):
                        self._emit({"kind": "plan_finished", "plan": plan})
                    else:
                        self._mark_remaining_skipped(plan)
                        self._emit({"kind": "plan_aborted",
                                    "reason": "没有可推进的任务（内部状态异常）"})
                    break

                # 等待任意一个任务完成（Ctrl+C 会在这里打断）
                finished = done_q.get()
                inflight.discard(finished)
                finished_task = plan.task(finished)
                failure = (finished_task
                           if finished_task is not None and finished_task.status == "failed"
                           else None)

                if failure is not None:
                    # 有任务失败：等其它在跑的任务自然收尾，再带着完整进度去重规划
                    while inflight:
                        inflight.discard(done_q.get())
                    if replans_left <= 0:
                        self._mark_remaining_skipped(plan)
                        self._emit({"kind": "plan_aborted",
                                    "reason": f"任务 {failure.id} 失败，且重规划次数已用完"})
                        break
                    new_plan, abort_reason = self.planner.replan(plan, failure)
                    replans_left -= 1
                    if new_plan is None:
                        self._mark_remaining_skipped(plan)
                        self._emit({"kind": "plan_aborted",
                                    "reason": abort_reason or "重规划器认为目标无法继续达成"})
                        break
                    plan.replace_remaining(new_plan.tasks)
                    order = validate_dag(plan)  # 合并后整体再校验（防御性）
                    self._emit({"kind": "plan_updated", "plan": plan})
                    continue

                # 全部终态才收工。必须同时满足：在跑窗口为空、完成队列已排空——
                # 否则可能把"已置为 failed 但还没被主循环处理"的任务的失败吞掉，
                # 提前 plan_finished 跳过重规划（竞态）。
                if (not inflight and done_q.empty()
                        and all(t.status in ("done", "skipped", "failed")
                                for t in plan.tasks)):
                    self._emit({"kind": "plan_finished", "plan": plan})
                    break
                # 未全部结束：回到顶部调度新解锁的任务
        except KeyboardInterrupt:
            # Ctrl+C：停止调度，在跑的任务标记失败并短暂等待收尾。
            # 线程是守护线程：单次模式进程退出时会被直接终止。
            self._cancelled = True
            for tid in list(inflight):
                task = plan.task(tid)
                if task is not None and task.status == "running":
                    task.status = "failed"
                    task.report = "被用户中断"
                    self._emit({"kind": "task_failed", "task": tid, "reason": task.report})
            for th in threading.enumerate():
                if th.name.startswith("yking-task-"):
                    th.join(timeout=5)
            self._mark_remaining_skipped(plan)
            self._emit({"kind": "plan_aborted", "reason": "被用户中断"})
            raise
        return plan

    # ------------------------------------------------------------------
    def _run_task(self, plan: Plan, task: PlanTask) -> None:
        task.status = "running"
        self._emit({"kind": "task_start", "task": task.id, "title": task.title})
        agent = Agent(
            self.llm,
            self.registry,
            self.ctx,
            system_prompt=self._executor_prompt(plan, task),
            max_steps=self.cfg.max_steps,
            on_event=lambda e, _tid=task.id: self._emit({**e, "task": _tid}),
            memory=self.memory,
        )
        try:
            report = agent.run_turn(self._task_brief(task))
        except KeyboardInterrupt:
            task.status = "failed"
            task.report = "被用户中断"
            if not self._cancelled:
                self._emit({"kind": "task_failed", "task": task.id, "reason": task.report})
            raise
        except Exception as e:
            task.status = "failed"
            task.report = f"执行异常: {type(e).__name__}: {e}"
            if not self._cancelled:
                self._emit({"kind": "task_failed", "task": task.id, "reason": task.report})
            return
        if self._cancelled:
            # 主线程已因 Ctrl+C 把本任务标记为失败，这里不再把结果写回
            return
        text = (report or "").strip()
        if text.upper().startswith("[FAILED]"):
            task.status = "failed"
            task.report = text[len("[FAILED]"):].lstrip(" :：") or "执行器报告任务失败"
            self._emit({"kind": "task_failed", "task": task.id, "reason": task.report})
        else:
            task.status = "done"
            task.report = text[:2000]
            self._emit({"kind": "task_done", "task": task.id})

    def _executor_prompt(self, plan: Plan, task: PlanTask) -> str:
        parts = [
            build_system_prompt(self.ctx),
            "",
            "# 当前处于 Plan-and-Execute 模式",
            "你是这个大计划中的任务执行器，只负责完成分配给你的子任务，不要去做其他任务的内容。",
            "",
            f"# 总体目标\n{plan.goal}",
            "",
            "# 计划全貌",
        ]
        for t in plan.tasks:
            icon = STATUS_STYLES.get(t.status, STATUS_STYLES["pending"])[0]
            parts.append(f"- [{icon}] {t.id}. {t.title}")
        dep_reports = []
        for d in task.deps:
            dt = plan.task(d)
            if dt is not None and dt.report:
                dep_reports.append(f"## 前置任务 {dt.id}. {dt.title} 的产出\n{dt.report}")
        if dep_reports:
            parts.append("")
            parts.append("# 依赖任务的产出（供参考）")
            parts.extend(dep_reports)
        if self.memory is not None:
            section = self.memory.render_section()
            if section:
                parts.append("")
                parts.append(section)
        return "\n".join(parts)

    @staticmethod
    def _task_brief(task: PlanTask) -> str:
        return (
            f"请完成任务 [{task.id}] {task.title}\n\n任务书：\n{task.goal}\n\n"
            "完成后在最终回答中总结：做了什么、关键文件/命令、验证结果。"
            "如果确实无法完成，最终回答以 [FAILED]: 开头并简要说明原因。"
        )

    @staticmethod
    def _mark_remaining_skipped(plan: Plan) -> None:
        for t in plan.tasks:
            if t.status in ("pending", "running"):
                t.status = "skipped"
                t.report = t.report or "计划中止"
