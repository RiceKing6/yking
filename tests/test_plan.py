"""Plan-and-Execute 冒烟测试（无需 API Key）。

覆盖：DAG 校验/拓扑排序、计划解析、replace_remaining、
完整流程（规划重试 → 按依赖执行 → 注入依赖产出 → 失败 → 重规划）、重规划成功（id 冲突重命名）。

运行: conda run -n yking python tests/test_plan.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yking.config import Config                                        # noqa: E402
from yking.plan import Plan, PlanError, PlanTask, parse_plan, validate_dag  # noqa: E402
from yking.planner import Planner, PlanExecutor                        # noqa: E402
from yking.tools import ToolContext, build_registry                    # noqa: E402


def submit_plan_call(args: dict, call_id: str = "p1") -> dict:
    return {"content": "", "usage": {},
            "tool_calls": [{"id": call_id, "name": "submit_plan",
                            "arguments": json.dumps(args, ensure_ascii=False)}]}


class FakeLLM:
    """按脚本依次返回预设响应，并记录每次请求。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, messages, tools, on_text_delta=None):
        self.calls.append((list(messages), tools))
        return self.script.pop(0)


def test_validate_dag() -> None:
    plan = Plan(goal="g", tasks=[
        PlanTask(id="1", title="a", goal="ga"),
        PlanTask(id="2", title="b", goal="gb", deps=["1"]),
        PlanTask(id="3", title="c", goal="gc", deps=["1", "2"]),
    ])
    assert validate_dag(plan) == ["1", "2", "3"]

    diamond = Plan(goal="g", tasks=[
        PlanTask(id="1", title="a", goal="ga"),
        PlanTask(id="2", title="b", goal="gb", deps=["1"]),
        PlanTask(id="3", title="c", goal="gc", deps=["1"]),
        PlanTask(id="4", title="d", goal="gd", deps=["2", "3"]),
    ])
    assert validate_dag(diamond) == ["1", "2", "3", "4"]

    cycle = Plan(goal="g", tasks=[
        PlanTask(id="1", title="a", goal="ga", deps=["2"]),
        PlanTask(id="2", title="b", goal="gb", deps=["1"]),
    ])
    try:
        validate_dag(cycle)
        raise AssertionError("应当检测到循环依赖")
    except PlanError as e:
        assert "循环依赖" in str(e)

    for bad in (
        Plan(goal="g", tasks=[PlanTask(id="1", title="a", goal="ga", deps=["9"])]),
        Plan(goal="g", tasks=[PlanTask(id="1", title="a", goal="ga", deps=["1"])]),
        Plan(goal="g", tasks=[PlanTask(id="1", title="a", goal="ga"),
                              PlanTask(id="1", title="b", goal="gb")]),
    ):
        try:
            validate_dag(bad)
            raise AssertionError("应当校验失败")
        except PlanError:
            pass
    print("  [ok] DAG 校验与拓扑排序（链式/菱形/环/未知依赖/自依赖/重复 id）")


def test_parse_plan() -> None:
    plan = parse_plan({
        "goal": "做一个demo",
        "tasks": [
            {"id": 1, "title": " t1 ", "goal": "g1", "deps": []},
            {"id": "2", "title": "t2", "goal": "g2", "deps": [1, 1, ""]},
        ],
    })
    assert plan.tasks[0].id == "1"
    assert plan.tasks[1].deps == ["1"]  # 数字 id 转字符串、去重、去空

    for bad_args in (
        {"goal": "g", "tasks": [{"id": "1", "title": "", "goal": "g", "deps": []}]},
        {"goal": "g", "tasks": [{"id": "1", "title": "t", "goal": "", "deps": []}]},
        {"goal": "g", "tasks": "not-a-list"},
        {"goal": "g", "tasks": [{"id": str(i), "title": "t", "goal": "g", "deps": []}
                                for i in range(13)]},  # 超过上限
    ):
        try:
            parse_plan(bad_args)
            raise AssertionError(f"应当解析失败: {bad_args}")
        except PlanError:
            pass
    print("  [ok] 计划解析（类型规整 / deps 清洗 / 字段与数量校验）")


def test_replace_remaining() -> None:
    plan = Plan(goal="g", tasks=[
        PlanTask(id="1", title="a", goal="ga"),
        PlanTask(id="2", title="b", goal="gb", deps=["1"]),
    ])
    plan.tasks[0].status = "done"
    new_tasks = [PlanTask(id="1", title="替代任务", goal="gx")]  # 与已完成任务的 id 冲突
    plan.replace_remaining(new_tasks)
    assert [t.id for t in plan.tasks] == ["1", "R1"]
    assert plan.task("R1").deps == []
    validate_dag(plan)
    print("  [ok] replace_remaining 保留已完成任务并自动重命名冲突 id")


def test_full_flow(root: Path) -> None:
    """规划重试 → 按依赖执行 → 依赖产出注入 → 任务失败 → 重规划 → 放弃。"""
    ctx = ToolContext(workspace=root, auto_approve=True)
    registry = build_registry(ctx)
    cfg = Config(api_key="dummy", workspace=root)

    good_plan_args = {
        "goal": "演示计划",
        "tasks": [
            {"id": "1", "title": "创建文件", "goal": "创建 hello.txt，内容为 A", "deps": []},
            {"id": "2", "title": "读取文件", "goal": "读取 hello.txt 并确认内容是 A", "deps": ["1"]},
            {"id": "3", "title": "验证", "goal": "运行命令验证文件存在", "deps": ["1", "2"]},
        ],
    }
    script = [
        # 规划第 1 次：纯文字不提交工具（测 nudge 重试路径）
        {"content": "好的，我来规划。", "tool_calls": [], "usage": {}},
        # 规划第 2 次：提交含未知依赖的坏计划（测校验反馈重试路径）
        submit_plan_call({"goal": "演示计划",
                          "tasks": [{"id": "1", "title": "a", "goal": "ga", "deps": ["9"]}]},
                         call_id="p_bad"),
        # 规划第 3 次：提交合法计划
        submit_plan_call(good_plan_args, call_id="p_good"),
        # 任务 1：调用工具写文件，然后给出产出
        {"content": "我来创建文件。", "usage": {"prompt_tokens": 1, "completion_tokens": 1},
         "tool_calls": [{"id": "e1", "name": "write_file",
                         "arguments": json.dumps({"path": "hello.txt", "content": "A"})}]},
        {"content": "已创建 hello.txt（内容 A）。", "tool_calls": [],
         "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        # 任务 2：直接完成
        {"content": "确认内容是 A。", "tool_calls": [],
         "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        # 任务 3：报告失败
        {"content": "[FAILED]: 验证命令在当前环境不可用", "tool_calls": [],
         "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        # 重规划：放弃
        submit_plan_call({"goal": "演示计划", "tasks": [],
                          "abort_reason": "验证环境不可用，目标无法达成"}, call_id="p_replan"),
    ]
    llm = FakeLLM(script)
    events = []
    planner = Planner(llm, on_event=events.append)
    plan = planner.create_plan("演示一下计划功能")
    assert [t.id for t in plan.tasks] == ["1", "2", "3"]
    # 重试路径生效：第 2 次被催促提交工具，第 3 次收到坏计划的错误反馈
    assert any("请立即通过调用 submit_plan" in (m.get("content") or "") for m in llm.calls[1][0])
    assert any("计划不合法" in (m.get("content") or "") for m in llm.calls[2][0])

    executor = PlanExecutor(llm, registry, ctx, cfg, on_event=events.append)
    plan = executor.run(plan)

    assert plan.task("1").status == "done"
    assert plan.task("2").status == "done"
    assert plan.task("3").status == "failed"
    assert (root / "hello.txt").read_text(encoding="utf-8") == "A"
    # 依赖任务的产出被注入到后续任务的上下文（任务 2 的系统提示里）
    task2_system = llm.calls[5][0][0]["content"]
    assert "前置任务 1" in task2_system and "hello.txt" in task2_system
    # 事件序列符合预期
    kinds = [e["kind"] for e in events]
    expected = (
        ["planning", "plan_created", "plan_started"]
        + ["task_start"]
        + ["step", "usage", "text", "tool_call", "tool_result", "step", "usage", "final"]
        + ["task_done"]
        + ["task_start"] + ["step", "usage", "final"] + ["task_done"]
        + ["task_start"] + ["step", "usage", "final"]
        + ["task_failed", "replanning", "plan_aborted"]
    )
    assert kinds == expected, kinds
    assert all(t.status != "pending" for t in plan.tasks)
    print("  [ok] 完整流程：规划重试 → 按依赖执行 → 注入依赖产出 → 失败 → 重规划 → 放弃")


def test_replan_success(root: Path) -> None:
    """任务失败后重规划出新任务接管，且新 id 与保留任务冲突时自动重命名 R1。"""
    ctx = ToolContext(workspace=root, auto_approve=True)
    registry = build_registry(ctx)
    cfg = Config(api_key="dummy", workspace=root)
    script = [
        submit_plan_call({"goal": "g", "tasks": [
            {"id": "1", "title": "会失败的任务", "goal": "g1", "deps": []},
            {"id": "2", "title": "依赖任务", "goal": "g2", "deps": ["1"]},
        ]}, call_id="p1"),
        {"content": "[FAILED]: 故意失败", "tool_calls": [],
         "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        # 重规划：新任务 id "1" 与已失败任务冲突 → 自动改名 R1
        submit_plan_call({"goal": "g", "tasks": [
            {"id": "1", "title": "替代方案", "goal": "换一种方式完成", "deps": []},
        ]}, call_id="p2"),
        {"content": "替代方案完成。", "tool_calls": [],
         "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
    ]
    llm = FakeLLM(script)
    events = []
    planner = Planner(llm, on_event=events.append)
    plan = planner.create_plan("演示重规划")
    executor = PlanExecutor(llm, registry, ctx, cfg, on_event=events.append)
    plan = executor.run(plan)

    assert plan.task("1").status == "failed"
    assert plan.task("2") is None  # 未完成任务被重规划替换
    r1 = plan.task("R1")
    assert r1 is not None and r1.status == "done"
    kinds = [e["kind"] for e in events]
    assert kinds == ["planning", "plan_created", "plan_started",
                     "task_start", "step", "usage", "final", "task_failed",
                     "replanning", "plan_updated", "task_start", "step", "usage",
                     "final", "task_done", "plan_finished"], kinds
    print("  [ok] 重规划成功：新任务接管（id 冲突自动重命名 R1）")


def main() -> int:
    print("yking Plan-and-Execute 冒烟测试:")
    test_validate_dag()
    test_parse_plan()
    test_replace_remaining()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        test_full_flow(root)
        test_replan_success(root)
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
