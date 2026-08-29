"""Plan-and-Execute：计划数据结构、DAG 校验与拓扑排序。"""
from __future__ import annotations

from dataclasses import dataclass, field

MAX_PLAN_TASKS = 12

# 步骤类型：research=只读调研 / code=编写修改文件 / command=执行命令
STEP_TYPES = ("research", "code", "command")

# 任务状态 -> (图标, rich 颜色, 中文标签)
STATUS_STYLES = {
    "pending": ("○", "dim", "待执行"),
    "running": ("●", "cyan", "执行中"),
    "done": ("✓", "green", "完成"),
    "failed": ("✗", "red", "失败"),
    "skipped": ("–", "yellow", "跳过"),
}


class PlanError(ValueError):
    """计划不合法。"""


@dataclass
class PlanTask:
    id: str
    title: str
    goal: str
    deps: list[str] = field(default_factory=list)  # 依赖的前置任务 id（DAG 的边）
    status: str = "pending"                        # pending/running/done/failed/skipped
    report: str = ""                               # 执行后的产出摘要，会喂给依赖它的任务
    type: str = "code"                             # 步骤类型（见 STEP_TYPES）
    acceptance: str = ""                           # 验收标准（Reviewer 逐条核验的判据）


@dataclass
class Plan:
    goal: str
    tasks: list[PlanTask] = field(default_factory=list)

    def task(self, task_id: str) -> PlanTask | None:
        return next((t for t in self.tasks if t.id == task_id), None)

    def replace_remaining(self, new_tasks: list[PlanTask]) -> None:
        """用重规划产出的任务替换所有未完成任务（保留已完成/已失败的记录）。

        若新任务 id 与保留任务冲突，自动重命名为 R1、R2…（deps 同步改写）。
        """
        kept = [t for t in self.tasks if t.status in ("done", "failed")]
        taken = {t.id for t in kept}
        if any(t.id in taken for t in new_tasks):
            mapping = {t.id: f"R{i}" for i, t in enumerate(new_tasks, 1)}
            for t in new_tasks:
                t.id = mapping[t.id]
                t.deps = [mapping.get(d, d) for d in t.deps]
        self.tasks = kept + list(new_tasks)


def parse_plan(args: dict) -> Plan:
    """把 submit_plan 工具的参数解析成 Plan 并做 DAG 校验。不合法抛 PlanError。"""
    if not isinstance(args, dict):
        raise PlanError("计划参数必须是 JSON 对象")
    goal = str(args.get("goal") or "").strip()
    raw_tasks = args.get("tasks")
    if not isinstance(raw_tasks, list):
        raise PlanError("tasks 必须是数组")
    if len(raw_tasks) > MAX_PLAN_TASKS:
        raise PlanError(f"任务数 {len(raw_tasks)} 超过上限 {MAX_PLAN_TASKS}，请合并拆分")
    tasks: list[PlanTask] = []
    for i, raw in enumerate(raw_tasks, 1):
        if not isinstance(raw, dict):
            raise PlanError(f"第 {i} 个任务不是 JSON 对象")
        task_id = str(raw.get("id", i)).strip() or str(i)
        title = str(raw.get("title") or "").strip()
        task_goal = str(raw.get("goal") or "").strip()
        if not title:
            raise PlanError(f"任务 {task_id} 缺少 title")
        if not task_goal:
            raise PlanError(f"任务 {task_id} 缺少 goal")
        deps: list[str] = []
        for d in raw.get("deps") or []:
            d = str(d).strip()
            if d and d not in deps:
                deps.append(d)
        step_type = str(raw.get("type") or "code").strip().lower()
        if step_type not in STEP_TYPES:
            raise PlanError(
                f"任务 {task_id} 的 type 非法: {step_type!r}（可选: {'/'.join(STEP_TYPES)}）")
        acceptance = str(raw.get("acceptance") or "").strip()
        tasks.append(PlanTask(id=task_id, title=title, goal=task_goal, deps=deps,
                              type=step_type, acceptance=acceptance))
    plan = Plan(goal=goal, tasks=tasks)
    validate_dag(plan)
    return plan


def validate_dag(plan: Plan) -> list[str]:
    """校验任务依赖图并返回拓扑序（执行顺序）。不合法抛 PlanError。"""
    ids = [t.id for t in plan.tasks]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise PlanError(f"任务 id 重复: {', '.join(dupes)}")
    id_set = set(ids)
    for t in plan.tasks:
        for d in t.deps:
            if d == t.id:
                raise PlanError(f"任务 {t.id} 依赖自身")
            if d not in id_set:
                raise PlanError(f"任务 {t.id} 依赖了不存在的任务 {d!r}")
    # Kahn 算法拓扑排序
    indegree = {t.id: len(t.deps) for t in plan.tasks}
    dependents: dict[str, list[str]] = {t.id: [] for t in plan.tasks}
    for t in plan.tasks:
        for d in t.deps:
            dependents[d].append(t.id)
    queue = [tid for tid in ids if indegree[tid] == 0]
    order: list[str] = []
    while queue:
        tid = queue.pop(0)
        order.append(tid)
        for nxt in dependents[tid]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if len(order) != len(plan.tasks):
        stuck = [t.id for t in plan.tasks if t.id not in order]
        raise PlanError(f"存在循环依赖，涉及任务: {', '.join(stuck)}")
    return order
